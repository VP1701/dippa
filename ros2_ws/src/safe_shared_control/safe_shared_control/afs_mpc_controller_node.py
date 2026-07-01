#!/usr/bin/env python3
"""
afs_mpc_controller_node.py -- ROS2 (Humble) CBF-MPC shared-control for AFS vehicle
==================================================================================
Replaces the reactive CBF filter with a horizon controller so the assist can
*steer* (not just brake) through corners. Same I/O as afs_controller_node.py, so
it drops into the same sim + RViz.

FORMULATION (agreed design)
  state q=[x_f,y_f,theta_f,gamma], input u=[v_f,omega], front-referenced model.
  - reference: held conditioned joystick command u_user over the horizon
               (minimal intervention; no goal/intent predictor)
  - model:     per-step LTV linearization re-rolled over a few SQP iterations
               (a single linearization at (q0,u_user) degenerates in the
               must-steer case: the held-straight nominal drives the discs INTO
               the obstacle, so the half-spaces are anchored at penetrating
               points and the QP can only brake.  Re-rolling the nominal around
               the improving plan fixes this.)  Each step linearized to A_k,B_k,c_k
               and augmented to incremental form  q~=[q;u_{k-1}], input du.
  - obstacles: per-cluster linearized half-spaces, discrete CBF-rate over horizon
               h_{k+1} >= (1-alpha) h_k - eps,  alpha = gamma_cbf*dt,  soft (slack)
  - terminal:  soft penalty on terminal speed v_f  (stoppability surrogate)
  - solver:    OSQP

  decision  Z = [ q~_1..q~_H , du_0..du_{H-1} , eps_1..eps_H ]   (dim 9H)

Subscribes: /right_controller/joy (Joy), /scan (LaserScan), /odom (Odometry),
            /afs/articulation (Float64)
Publishes:  /afs/cmd (Float64MultiArray [v_f,omega]), /afs/ogm (OccupancyGrid),
            /afs/intent_path (Path, held-command rollout),
            /afs/planned_path (Path, MPC solution)
"""
import math
import numpy as np
from scipy import ndimage, sparse
import osqp


# =========================================================================== #
# Vehicle model (ROS-independent)                                             #
# =========================================================================== #
class AFSModel:
    def __init__(self, L_f, L_r, r_disc):
        self.L_f, self.L_r, self.r_disc = L_f, L_r, r_disc
        # (kind, offset) for each disc: 3 along front body, 2 along rear body
        self.discs = [('f', L_f / 2),('r', L_r / 2)]

    def f(self, q, u, dt):
        """Nonlinear one-step (no wrap/clip; limits handled by QP constraints)."""
        xf, yf, th, g = q
        vf, om = u
        Dq = self.L_f * math.cos(g) + self.L_r
        thd = (self.L_r * om + vf * math.sin(g)) / Dq
        return np.array([xf + dt * vf * math.cos(th),
                         yf + dt * vf * math.sin(th),
                         th + dt * thd,
                         g + dt * om])

    def linearize(self, q, u, dt, eps=1e-6):
        """A=df/dq, B=df/du, c=f-Aq-Bu via central differences (robust)."""
        A = np.zeros((4, 4)); B = np.zeros((4, 2))
        for j in range(4):
            dq = np.zeros(4); dq[j] = eps
            A[:, j] = (self.f(q + dq, u, dt) - self.f(q - dq, u, dt)) / (2 * eps)
        for j in range(2):
            du = np.zeros(2); du[j] = eps
            B[:, j] = (self.f(q, u + du, dt) - self.f(q, u - du, dt)) / (2 * eps)
        c = self.f(q, u, dt) - A @ q - B @ u
        return A, B, c

    def disc_pos(self, q):
        xf, yf, th, g = q
        thr = th - g
        F = np.array([xf, yf])
        ef = np.array([math.cos(th), math.sin(th)])
        er = np.array([math.cos(thr), math.sin(thr)])
        P = F - self.L_f * ef
        out = []
        for kind, off in self.discs:
            out.append(F - off * ef if kind == 'f' else P - off * er)
        return np.array(out)                       # (5,2)

    def disc_jac(self, q):
        """J_i = d p_i / d q  (2x4) per disc, evaluated at q."""
        xf, yf, th, g = q
        thr = th - g
        s_th, c_th = math.sin(th), math.cos(th)
        s_tr, c_tr = math.sin(thr), math.cos(thr)
        Js = []
        for kind, off in self.discs:
            if kind == 'f':
                J = np.array([[1.0, 0.0, off * s_th, 0.0],
                              [0.0, 1.0, -off * c_th, 0.0]])
            else:
                J = np.array([[1.0, 0.0, self.L_f * s_th + off * s_tr, -off * s_tr],
                              [0.0, 1.0, -self.L_f * c_th - off * c_tr, off * c_tr]])
            Js.append(J)
        return Js


# =========================================================================== #
# CBF-MPC solver (ROS-independent)                                            #
# =========================================================================== #
class AFSMPC:
    def __init__(self, model, H=8, dt=0.2, gamma_cbf=2.5, margin=0.1,
                 influence_R=1.5, q_v=1.0, q_w=0.3, r_v=0.1, r_w=0.25,
                 w_term=1.0, w_slack=1e4, w_slack_lin=1e3,
                 v_max=1.2, omega_max=1.2, g_max=0.75,
                 sqp_iters=3, intent_reset_thresh=1.0, eps_tol=0.03):
        self.m = model
        self.H, self.dt = H, dt
        self.alpha = float(np.clip(gamma_cbf * dt, 1e-3, 1.0))
        self.margin, self.influence_R = margin, influence_R
        self.Qu = np.array([q_v, q_w]); self.R = np.array([r_v, r_w])
        self.w_term, self.w_slack = w_term, w_slack
        self.w_slack_lin = w_slack_lin
        self.v_max, self.omega_max, self.g_max = v_max, omega_max, g_max
        self.eps_tol = float(eps_tol)     # first-step slack above this => "no safe action"
        self.nz = 9 * H
        self.sqp_iters = int(sqp_iters)
        self.intent_reset_thresh = float(intent_reset_thresh)
        self.x_prev = None
        self._useq_prev = None
        self._u_user_prev = None

    # --- index helpers into Z ---
    def iq(self, k):  return 6 * (k - 1)            # q~_k start (k=1..H)
    def idu(self, k): return 6 * self.H + 2 * k     # du_k start (k=0..H-1)
    def ieps(self, k): return 8 * self.H + (k - 1)  # eps_k      (k=1..H)

    def min_clearance(self, q, clusters):
        """Smallest disc clearance h = dist - r_disc - margin at pose q (h<0 => in keep-out)."""
        hmin = np.inf
        for p in self.m.disc_pos(q):
            for wpts in clusters:
                dmin = float(np.sqrt(((p[None, :] - wpts) ** 2).sum(1)).min())
                hmin = min(hmin, dmin - self.m.r_disc - self.margin)
        return float(hmin)

    def solve(self, q0, u_prev, u_user, clusters):
        H, dt = self.H, self.dt
        q0 = np.asarray(q0, float); u_prev = np.asarray(u_prev, float)
        u_user = np.asarray(u_user, float)
        # Warm-start the SQP nominal from the previous plan for fast, stable
        # convergence and a committed avoidance direction -- EXCEPT when the
        # driver's intent flips (forward<->reverse or a large command jump),
        # where the stale plan makes the controller sticky and it crawls toward
        # the new command. On a flip we reseed the nominal from the held command
        # so the new intent takes effect immediately.
        prev = self._u_user_prev
        flip = (prev is None
                or u_user[0] * prev[0] < -0.05
                or np.linalg.norm(u_user - prev) > self.intent_reset_thresh)
        if (self._useq_prev is not None) and (not flip):
            u_seq = np.vstack([self._useq_prev[1:], self._useq_prev[-1:]])
        else:
            u_seq = np.tile(u_user, (H, 1))
        self._u_user_prev = np.array(u_user, float)
        x_ws = self.x_prev
        best = None
        for _ in range(self.sqp_iters):
            qbar = [q0]
            for k in range(H):
                qbar.append(self.m.f(qbar[k], u_seq[k], dt))      # nonlinear nominal
            ATs, BTs, cTs = [], [], []
            for k in range(H):                                    # per-step (LTV)
                A, B, c = self.m.linearize(qbar[k], u_seq[k], dt)
                ATs.append(np.block([[A, B], [np.zeros((2, 4)), np.eye(2)]]))
                BTs.append(np.vstack([B, np.eye(2)]))
                cTs.append(np.concatenate([c, np.zeros(2)]))
            x, ok = self._qp(q0, u_prev, u_user, qbar, ATs, BTs, cTs, clusters, x_ws)
            if not ok:
                break
            x_ws = x
            u_seq = np.array([x[self.iq(k + 1) + 4:self.iq(k + 1) + 6] for k in range(H)])
            planned = [q0] + [x[self.iq(k):self.iq(k) + 4] for k in range(1, H + 1)]
            best = (u_prev + x[self.idu(0):self.idu(0) + 2], planned, list(qbar), x, u_seq)
        if best is None:
            # SAFE FALLBACK: a failed/maxed-out solve must NOT hold the last
            # (possibly forward) command -- that would drive through a wall.
            # Stop instead; the next tick re-solves from rest.
            self._useq_prev = None
            info = {'min_h_now': self.min_clearance(q0, clusters),
                    'slack_now': float('inf'), 'slack_max': float('inf'),
                    'infeasible': True, 'solver_failed': True}
            return np.zeros(2), None, [q0], False, info
        u0, planned, qbar, x, u_seq = best
        self.x_prev = x; self._useq_prev = u_seq
        slack_now = float(x[self.ieps(1)]) if self.H >= 1 else 0.0
        slack_max = float(max(x[self.ieps(k)] for k in range(1, self.H + 1)))
        info = {'min_h_now': self.min_clearance(q0, clusters),
                'slack_now': slack_now, 'slack_max': slack_max,
                'infeasible': slack_now > self.eps_tol, 'solver_failed': False}
        return u0, planned, qbar, True, info

    def _qp(self, q0, u_prev, u_user, qbar, ATs, BTs, cTs, clusters, x_ws):
        H, dt, nz = self.H, self.dt, self.nz
        qt0 = np.concatenate([q0, u_prev])
        # ---- cost (diagonal P, linear q) ----
        Pd = np.full(nz, 1e-6); qv = np.zeros(nz)
        for k in range(H):
            iu = self.iq(k + 1) + 4
            Pd[iu] += 2 * self.Qu[0]; qv[iu] += -2 * self.Qu[0] * u_user[0]
            Pd[iu + 1] += 2 * self.Qu[1]; qv[iu + 1] += -2 * self.Qu[1] * u_user[1]
        for k in range(H):
            idu = self.idu(k); Pd[idu] += 2 * self.R[0]; Pd[idu + 1] += 2 * self.R[1]
        Pd[self.iq(H) + 4] += 2 * self.w_term
        for k in range(1, H + 1):
            Pd[self.ieps(k)] += 2 * self.w_slack
            qv[self.ieps(k)] += self.w_slack_lin     # exact-penalty term: don't cut the margin cheaply
        P = sparse.diags(Pd).tocsc()
        rows, cols, data, lo, up = [], [], [], [], []
        r = 0
        def add(rr, cc, vv): rows.append(rr); cols.append(cc); data.append(vv)
        # dynamics equalities (per-step matrices)
        for k in range(H):
            iqp = self.iq(k + 1); idu = self.idu(k); At, Bt, ct = ATs[k], BTs[k], cTs[k]
            for a in range(6):
                add(r + a, iqp + a, 1.0)
                for b in range(2): add(r + a, idu + b, -Bt[a, b])
            if k == 0:
                rhs = At @ qt0 + ct
            else:
                iqk = self.iq(k)
                for a in range(6):
                    for b in range(6): add(r + a, iqk + b, -At[a, b])
                rhs = ct
            for a in range(6): lo.append(rhs[a]); up.append(rhs[a])
            r += 6
        # obstacle CBF-rate (per disc, per cluster)
        Pn = [self.m.disc_pos(qbar[k]) for k in range(H + 1)]
        Jn = [self.m.disc_jac(qbar[k]) for k in range(H + 1)]
        aa = 1.0 - self.alpha
        for i in range(len(self.m.discs)):
            pi = np.array([Pn[k][i] for k in range(H + 1)])
            for wpts in clusters:
                d2 = ((pi[:, None, :] - wpts[None, :, :]) ** 2).sum(-1)
                kb = d2.argmin(1); cpts = wpts[kb]          # single closest cell
                dist = np.sqrt(d2[np.arange(H + 1), kb])
                if dist.min() > self.influence_R:
                    continue
                diff = pi - cpts
                nrm = diff / np.clip(dist[:, None], 1e-9, None)
                hbar = dist - self.m.r_disc - self.margin
                avec = np.array([nrm[k] @ Jn[k][i] for k in range(H + 1)])
                bk = hbar - np.array([avec[k] @ qbar[k] for k in range(H + 1)])
                for k in range(H):
                    iqp = self.iq(k + 1)
                    for b in range(4): add(r, iqp + b, avec[k + 1][b])
                    add(r, self.ieps(k + 1), 1.0)
                    if k == 0:
                        l = aa * hbar[0] - bk[1]
                    else:
                        iqk = self.iq(k)
                        for b in range(4): add(r, iqk + b, -aa * avec[k][b])
                        l = aa * bk[k] - bk[k + 1]
                    lo.append(l); up.append(np.inf); r += 1
        # box constraints
        for k in range(H):
            iu = self.iq(k + 1) + 4
            add(r, iu, 1.0); lo.append(-self.v_max); up.append(self.v_max); r += 1
            add(r, iu + 1, 1.0); lo.append(-self.omega_max); up.append(self.omega_max); r += 1
        for k in range(1, H + 1):
            add(r, self.iq(k) + 3, 1.0); lo.append(-self.g_max); up.append(self.g_max); r += 1
        for k in range(1, H + 1):
            add(r, self.ieps(k), 1.0); lo.append(0.0); up.append(np.inf); r += 1
        Ac = sparse.csc_matrix((data, (rows, cols)), shape=(r, nz))
        prob = osqp.OSQP()
        prob.setup(P=P, q=qv, A=Ac, l=np.array(lo), u=np.array(up), verbose=False,
                   warm_starting=True, max_iter=8000, eps_abs=1e-4, eps_rel=1e-4,
                   polish=True)
        if x_ws is not None and len(x_ws) == nz:
            try: prob.warm_start(x=x_ws)
            except Exception: pass
        res = prob.solve()
        ok = res.info.status_val in (1, 2) and res.x is not None and np.all(np.isfinite(res.x))
        return (res.x if ok else None), ok


def clip_halfplane(poly, n, d):
    """Clip convex polygon (list of (x,y)) to the half-plane n.x >= d."""
    out = []
    N = len(poly)
    for i in range(N):
        a = poly[i]; b = poly[(i + 1) % N]
        sa = n[0] * a[0] + n[1] * a[1] - d
        sb = n[0] * b[0] + n[1] * b[1] - d
        if sa >= 0:
            out.append(a)
        if (sa > 0) != (sb > 0):
            t = sa / (sa - sb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def step_color(k, H):
    """Strong green (near) -> yellow -> red (far) ramp over horizon step k."""
    s = k / max(1, H - 1)
    return (2.0 * s, 1.0, 0.0) if s < 0.5 else (1.0, 2.0 * (1.0 - s), 0.0)


def build_clusters(prob_grid, res, origin, nx, ny, region_min, region_max,
                   occ_thresh=0.6):
    occupied = prob_grid > occ_thresh
    i0 = max(0, int((region_min[0] - origin[0]) / res))
    j0 = max(0, int((region_min[1] - origin[1]) / res))
    i1 = min(nx, int((region_max[0] - origin[0]) / res) + 1)
    j1 = min(ny, int((region_max[1] - origin[1]) / res) + 1)
    if i1 <= i0 or j1 <= j0:
        return []
    local = occupied[j0:j1, i0:i1]
    if not local.any():
        return []
    labels, n = ndimage.label(local)
    clusters = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        clusters.append(np.stack([origin[0] + (xs + i0 + 0.5) * res,
                                   origin[1] + (ys + j0 + 0.5) * res], axis=1))
    return clusters


# =========================================================================== #
# ROS2 node                                                                   #
# =========================================================================== #
def _ros():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float64, Float64MultiArray, ColorRGBA
    from sensor_msgs.msg import LaserScan, Joy
    from nav_msgs.msg import Odometry, OccupancyGrid, Path
    from geometry_msgs.msg import PoseStamped, Quaternion, Point
    from visualization_msgs.msg import Marker, MarkerArray
    return (rclpy, Node, qos_profile_sensor_data, Float64, Float64MultiArray,
            LaserScan, Joy, Odometry, OccupancyGrid, Path, PoseStamped, Quaternion,
            Marker, MarkerArray, Point, ColorRGBA)


class LocalOGM:
    def __init__(self, width, height, res, origin=(0.0, 0.0)):
        self.res = res; self.origin = np.array(origin, float)
        self.nx, self.ny = int(round(width / res)), int(round(height / res))
        self.logodds = np.zeros((self.ny, self.nx))
        self.L_OCC, self.L_FREE, self.L_CLAMP = 0.85, -0.40, 6.0

    def prob(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds))

    def update_from_scan(self, sx, sy, syaw, rel, ranges, range_max):
        step = self.res * 0.5; n_steps = max(1, int(range_max / step))
        ab = syaw + np.asarray(rel)
        dirs = np.stack([np.cos(ab), np.sin(ab)], 1)
        s = np.arange(1, n_steps + 1) * step
        pts = np.array([sx, sy])[None, None, :] + s[None, :, None] * dirs[:, None, :]
        ix = ((pts[..., 0] - self.origin[0]) / self.res).astype(np.intp)
        iy = ((pts[..., 1] - self.origin[1]) / self.res).astype(np.intp)
        inb = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        r = np.asarray(ranges, float)
        free = (s[None, :] < (r[:, None] - self.res)) & inb
        hit = r < (range_max - 1e-3)
        hs = np.clip((r / step).astype(np.intp) - 1, 0, n_steps - 1)
        hm = hit[:, None] & (np.arange(n_steps)[None, :] == hs[:, None]) & inb
        np.add.at(self.logodds, (iy[free], ix[free]), self.L_FREE)
        np.add.at(self.logodds, (iy[hm], ix[hm]), self.L_OCC)
        np.clip(self.logodds, -self.L_CLAMP, self.L_CLAMP, out=self.logodds)


def quat_to_yaw(qz, qw):
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)


def main(args=None):
    (rclpy, Node, qos_sensor, Float64, Float64MultiArray, LaserScan, Joy,
     Odometry, OccupancyGrid, Path, PoseStamped, Quaternion,
     Marker, MarkerArray, Point, ColorRGBA) = _ros()

    class AFSMPCNode(Node):
        def __init__(self):
            super().__init__("afs_mpc_controller")
            p = self.declare_parameter
            self.L_f = p("link_front", 0.5).value
            self.L_r = p("link_rear", 0.5).value
            self.r_disc = p("disc_radius", 0.4).value
            self.g_max = p("gamma_max", 0.75).value
            self.v_max = p("v_max", 1.2).value
            self.omega_max = p("omega_max", 1.2).value
            self.H = p("horizon", 8).value
            self.dt = p("mpc_dt", 0.2).value
            self.gamma_cbf = p("cbf_gamma", 0.5).value
            self.margin = p("margin", 0.1).value
            self.influence_R = p("influence_radius", 1.5).value
            self.viz_polytopes = p("viz_polytopes", True).value
            self.viz_disc = p("viz_disc", 0).value      # which disc's polytope to draw (0=front axle)
            self.q_v = p("track_v", 1.0).value
            self.q_w = p("track_w", 0.3).value
            self.r_v = p("smooth_v", 0.1).value
            self.r_w = p("smooth_w", 0.25).value
            self.w_term = p("terminal_speed_weight", 1.0).value
            self.w_slack = p("slack_weight", 1e4).value
            self.w_slack_lin = p("slack_weight_lin", 1e3).value
            self.sqp_iters = p("sqp_iters", 3).value
            self.intent_reset_thresh = p("intent_reset_thresh", 1.0).value
            self.stop_on_infeasible = p("stop_on_infeasible", True).value
            self.eps_tol = p("infeasible_slack_tol", 0.03).value
            self.ctrl_rate = p("control_rate", 10.0).value
            ow = p("ogm_width", 12.0).value; oh = p("ogm_height", 8.0).value
            ores = p("ogm_res", 0.06).value; oorigin = p("ogm_origin", [0.0, 0.0]).value
            self.joy_topic = p("joy_topic", "/right_controller/joy").value
            self.axis_drive = p("axis_drive", 0).value
            self.axis_steer = p("axis_steer", 1).value
            self.invert_drive = p("invert_drive", False).value
            self.invert_steer = p("invert_steer", True).value
            self.deadzone = p("deadzone", 0.06).value
            self.lpf_alpha = p("joy_lpf_alpha", 0.6).value    # light conditioning; MPC du-cost does the smoothing
            self.joy_timeout = p("joy_timeout", 0.5).value

            self.model = AFSModel(self.L_f, self.L_r, self.r_disc)
            self.mpc = AFSMPC(self.model, H=self.H, dt=self.dt, gamma_cbf=self.gamma_cbf,
                              margin=self.margin, influence_R=self.influence_R,
                              q_v=self.q_v, q_w=self.q_w, r_v=self.r_v, r_w=self.r_w,
                              w_term=self.w_term, w_slack=self.w_slack,
                              w_slack_lin=self.w_slack_lin,
                              v_max=self.v_max, omega_max=self.omega_max, g_max=self.g_max,
                              sqp_iters=self.sqp_iters,
                              intent_reset_thresh=self.intent_reset_thresh,
                              eps_tol=self.eps_tol)
            self.ogm = LocalOGM(ow, oh, ores, oorigin)

            self.state = None; self.pose = None; self.gamma = 0.0
            self.v_cmd = 0.0; self.omega_cmd = 0.0
            self.u_prev = np.zeros(2); self._last_joy = None

            self.pub = self.create_publisher(Float64MultiArray, "afs/cmd", 10)
            self.pub_map = self.create_publisher(OccupancyGrid, "afs/ogm", 1)
            self.pub_safe = self.create_publisher(OccupancyGrid, "afs/safe_region", 1)
            self.pub_intent = self.create_publisher(Path, "afs/intent_path", 1)
            self.pub_plan = self.create_publisher(Path, "afs/planned_path", 1)
            self.pub_poly = self.create_publisher(MarkerArray, "afs/polytopes", 1)
            self.pub_status = self.create_publisher(MarkerArray, "afs/status", 1)
            self.create_subscription(Joy, self.joy_topic, self.joy_cb, 10)
            self.create_subscription(LaserScan, "scan", self.scan_cb, qos_sensor)
            self.create_subscription(Odometry, "odom", self.odom_cb, 10)
            self.create_subscription(Float64, "afs/articulation", self.art_cb, 10)
            self.create_timer(1.0 / self.ctrl_rate, self.control_tick)
            self.create_timer(0.2, self.publish_map)
            self.create_timer(0.2, self.publish_safe_region)
            self.get_logger().info(f"AFS CBF-MPC ready (H={self.H}, dt={self.dt}).")

        def _dz(self, x):
            if abs(x) < self.deadzone: return 0.0
            return (x - math.copysign(self.deadzone, x)) / (1.0 - self.deadzone)

        def joy_cb(self, msg):
            self._last_joy = self.get_clock().now()
            na = len(msg.axes)
            if na > self.axis_drive and na > self.axis_steer:
                dr = self._dz(msg.axes[self.axis_drive]) * (-1.0 if self.invert_drive else 1.0)
                st = self._dz(msg.axes[self.axis_steer]) * (-1.0 if self.invert_steer else 1.0)
                vt = float(np.clip(dr, -1, 1) * self.v_max)
                wt = float(np.clip(st, -1, 1) * self.omega_max)
                a = self.lpf_alpha                            # low-pass = light intent conditioning
                self.v_cmd = (1 - a) * self.v_cmd + a * vt
                self.omega_cmd = (1 - a) * self.omega_cmd + a * wt

        def odom_cb(self, msg):
            self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                         quat_to_yaw(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
            self._refresh()

        def art_cb(self, msg):
            self.gamma = float(msg.data); self._refresh()

        def _refresh(self):
            if self.pose is not None:
                self.state = np.array([self.pose[0], self.pose[1], self.pose[2], self.gamma])

        def scan_cb(self, msg):
            if self.state is None: return
            xf, yf, th, g = self.state
            rel = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
            ranges = np.where(np.isfinite(msg.ranges), msg.ranges, msg.range_max)
            self.ogm.update_from_scan(xf, yf, th, rel, ranges, msg.range_max)

        def control_tick(self):
            if self.state is None: return
            if self._last_joy is None or \
               (self.get_clock().now() - self._last_joy).nanoseconds * 1e-9 > self.joy_timeout:
                self.v_cmd = 0.0; self.omega_cmd = 0.0
            u_user = np.array([self.v_cmd, self.omega_cmd])
            # clusters in a window around the vehicle
            cx, cy = self.state[0], self.state[1]
            reach = self.v_max * self.H * self.dt + self.influence_R + self.L_f + self.L_r
            clusters = build_clusters(self.ogm.prob(), self.ogm.res, self.ogm.origin,
                                      self.ogm.nx, self.ogm.ny,
                                      (cx - reach, cy - reach), (cx + reach, cy + reach))
            u0, planned, qbar, ok, info = self.mpc.solve(self.state, self.u_prev, u_user, clusters)
            infeasible = bool(info['infeasible'])         # no safe action for the next step
            unsafe_now = info['min_h_now'] < 0.0          # a disc is already inside the keep-out
            driver_cmd = bool(abs(u_user[0]) > 1e-3 or abs(u_user[1]) > 1e-3)
            if infeasible and self.stop_on_infeasible:
                u0 = np.zeros(2)                          # STOP: refuse to enter the unsafe region
            self.u_prev = u0
            self.pub.publish(Float64MultiArray(data=[float(u0[0]), float(u0[1])]))
            # intent = nonlinear rollout of the *held user command* (unaided path)
            intent = [self.state]
            for _ in range(self.H):
                intent.append(self.model.f(intent[-1], u_user, self.dt))
            self._publish_path(self.pub_intent, intent)
            if planned is not None:
                self._publish_path(self.pub_plan, planned)
            self._publish_status(infeasible, unsafe_now, driver_cmd, info)
            if self.viz_polytopes and planned is not None:
                self._publish_polytopes(planned, clusters)

        def _publish_path(self, pub, traj):
            path = Path(); path.header.frame_id = "odom"
            path.header.stamp = self.get_clock().now().to_msg()
            for q in traj:
                ps = PoseStamped(); ps.header = path.header
                ps.pose.position.x = float(q[0]); ps.pose.position.y = float(q[1])
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            pub.publish(path)

        def _publish_polytopes(self, traj, clusters):
            # For the chosen disc, at each predicted horizon position, intersect the
            # tangent half-planes  n.(x - c) >= r_disc + margin  from every nearby
            # obstacle into the convex feasible pocket for that disc CENTRE, and draw
            # it as a polygon coloured by horizon step (green=now -> red=far). When a
            # pocket pinches to nothing the step is skipped -> that gap is where the
            # controller runs out of feasible room (i.e. why it brakes).
            arr = MarkerArray()
            clear = Marker(); clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            r_m = self.r_disc + self.margin
            R = self.influence_R
            i = int(self.viz_disc)
            H = len(traj)
            for k, q in enumerate(traj):
                p = self.model.disc_pos(q)[i]
                poly = [(p[0] - R, p[1] - R), (p[0] + R, p[1] - R),
                        (p[0] + R, p[1] + R), (p[0] - R, p[1] + R)]
                for wpts in clusters:
                    dd = np.sqrt(((p[None, :] - wpts) ** 2).sum(1))
                    if dd.min() > R:
                        continue
                    c = wpts[dd.argmin()]
                    nrm = p - c; nn = float(np.linalg.norm(nrm))
                    if nn < 1e-9:
                        continue
                    nrm = nrm / nn
                    d = float(nrm[0] * c[0] + nrm[1] * c[1] + r_m)   # keep n.x >= d
                    poly = clip_halfplane(poly, nrm, d)
                    if len(poly) < 3:
                        break
                if len(poly) < 3:
                    continue                                   # pocket pinched shut
                col = step_color(k, H)
                m = Marker()
                m.header.frame_id = "odom"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "polytope"; m.id = k
                m.type = Marker.LINE_STRIP; m.action = Marker.ADD
                m.scale.x = 0.02
                m.color = ColorRGBA(r=float(col[0]), g=float(col[1]), b=float(col[2]), a=0.85)
                m.pose.orientation.w = 1.0
                for vx, vy in list(poly) + [poly[0]]:
                    m.points.append(Point(x=float(vx), y=float(vy), z=0.02))
                arr.markers.append(m)
            self.pub_poly.publish(arr)

        def _publish_status(self, infeasible, unsafe_now, driver_cmd, info):
            # Two-state driver indicator above the vehicle:
            #   RED    - no safe action exists (QP needed > eps_tol slack); machine stopped.
            #            Driver should change intent (e.g. reverse) to find a feasible action.
            #   ORANGE - a disc is already inside the keep-out (h<0); recovering.
            #   GREEN  - OK / clear.
            x, y = float(self.state[0]), float(self.state[1])
            if infeasible:
                col = (1.0, 0.1, 0.1)
                txt = "NO SAFE ACTION - STOPPED" + ("  (try reversing)" if driver_cmd else "")
            elif unsafe_now:
                col = (1.0, 0.55, 0.0)
                txt = "INSIDE KEEP-OUT  h=%.2f m" % info['min_h_now']
            else:
                col = (0.1, 0.9, 0.1)
                txt = "OK  h=%.2f m  slack=%.03f" % (info['min_h_now'], info.get('slack_now', 0.0))
            arr = MarkerArray()
            light = Marker()
            light.header.frame_id = "odom"
            light.header.stamp = self.get_clock().now().to_msg()
            light.ns = "status_light"; light.id = 0
            light.type = Marker.SPHERE; light.action = Marker.ADD
            light.pose.position.x = x; light.pose.position.y = y; light.pose.position.z = 1.4
            light.pose.orientation.w = 1.0
            light.scale.x = light.scale.y = light.scale.z = 0.45
            light.color = ColorRGBA(r=float(col[0]), g=float(col[1]), b=float(col[2]), a=0.95)
            arr.markers.append(light)
            label = Marker()
            label.header = light.header
            label.ns = "status_text"; label.id = 1
            label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD
            label.pose.position.x = x; label.pose.position.y = y; label.pose.position.z = 1.9
            label.pose.orientation.w = 1.0
            label.scale.z = 0.35
            label.color = ColorRGBA(r=float(col[0]), g=float(col[1]), b=float(col[2]), a=1.0)
            label.text = txt
            arr.markers.append(label)
            self.pub_status.publish(arr)

        def _grid_msg(self, data_int8):
            m = OccupancyGrid(); m.header.frame_id = "odom"
            m.header.stamp = self.get_clock().now().to_msg()
            m.info.resolution = float(self.ogm.res)
            m.info.width = self.ogm.nx; m.info.height = self.ogm.ny
            m.info.origin.position.x = float(self.ogm.origin[0])
            m.info.origin.position.y = float(self.ogm.origin[1])
            m.info.origin.orientation.w = 1.0
            m.data = data_int8.flatten().tolist()
            return m

        def publish_map(self):
            prob = self.ogm.prob()
            data = np.full(prob.shape, -1, dtype=np.int8)
            known = np.abs(self.ogm.logodds) > 1e-6
            data[known] = (prob[known] * 100).astype(np.int8)
            self.pub_map.publish(self._grid_msg(data))

        def publish_safe_region(self):
            # Safe region = free space minus obstacles inflated by (r_disc+margin).
            # A disc CENTRE in a '0' cell keeps that disc at least `margin` clear
            # of any obstacle; '100' cells are the keep-out. The vehicle is safe
            # iff all five disc centres lie in '0'. If a corridor's free channel
            # vanishes here, it is too narrow to thread at this margin/radius.
            prob = self.ogm.prob()
            occ = prob > 0.6
            known = np.abs(self.ogm.logodds) > 1e-6
            rad = int(np.ceil((self.r_disc + self.margin) / self.ogm.res))
            if rad >= 1 and occ.any():
                yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
                disk = (xx * xx + yy * yy) <= rad * rad
                keepout = ndimage.binary_dilation(occ, structure=disk)
            else:
                keepout = occ
            data = np.full(prob.shape, -1, dtype=np.int8)
            data[known & ~keepout] = 0       # safe for a disc centre
            data[keepout] = 100              # inflated keep-out
            self.pub_safe.publish(self._grid_msg(data))

    rclpy.init(args=args)
    node = AFSMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
