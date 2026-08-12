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

Subscribes: /right_controller/joy (Joy), /map (OccupancyGrid -- static ground
            truth, received once from the sim and latched via TRANSIENT_LOCAL
            QoS; there is no incremental mapping any more), /odom (Odometry),
            /afs/articulation (Float64)
Publishes:  /afs/cmd (Float64MultiArray [v_f,omega]), /afs/ogm (OccupancyGrid,
            the received map echoed back for RViz),
            /afs/intent_path (Path, held-command rollout),
            /afs/planned_path (Path, MPC solution)
"""
import math
import time

import numpy as np
from scipy import ndimage

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       QoSHistoryPolicy)

from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Float64, Float64MultiArray, ColorRGBA, Empty
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import (Buffer, TransformListener, LookupException,
                     ConnectivityException, ExtrapolationException)

from safe_shared_control.afs_mpc import AFSMPC
from safe_shared_control.afs_model import AFSModel

class StaticOGM:
    """Wraps a nav_msgs/OccupancyGrid message received once from /map with the
    small interface build_clusters()/control_tick() expect (prob(), known(),
    res, origin, nx, ny). There is no incremental update any more -- the
    obstacle layout is fully known up front."""

    def __init__(self, unknown_is_obstacle=True):
        self.res = None
        self.origin = None
        self.nx = self.ny = 0
        self._prob = None
        self._known = None
        self.unknown_is_obstacle = bool(unknown_is_obstacle)

        

    def ready(self):
        return self._prob is not None

    def load_from_msg(self, msg):
        self.res = float(msg.info.resolution)
        self.origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y], float)
        self.nx, self.ny = msg.info.width, msg.info.height
        data = np.frombuffer(msg.data, dtype=np.int8).reshape((self.ny, self.nx))
        self._known = data >= 0
        fill = 1.0 if self.unknown_is_obstacle else 0.0
        self._prob = np.where(self._known, data / 100.0, fill)

    def prob(self):
        return self._prob

    def known(self):
        return self._known


def quat_to_yaw(qz, qw):
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)

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
    i0 = max(0, int((region_min[0] - origin[0]) / res))
    j0 = max(0, int((region_min[1] - origin[1]) / res))
    i1 = min(nx, int((region_max[0] - origin[0]) / res) + 1)
    j1 = min(ny, int((region_max[1] - origin[1]) / res) + 1)
    if i1 <= i0 or j1 <= j0:
        return []
    local = prob_grid[j0:j1, i0:i1] > occ_thresh 
    # Close 1-cell gaps: dilate then erode. Lidar sparsity leaves pinholes in
    # walls, which split one obstacle into many clusters and multiply the
    # constraint rows for no physical reason.
    local = ndimage.binary_closing(local, structure=np.ones((3, 3)))
    # Keep only the boundary: a cell in an obstacle's interior can never be the
    # nearest point to a disc outside it, so it contributes nothing to any
    # constraint while still being scanned. ~5x fewer points on solid regions.
    local = local & ~ndimage.binary_erosion(local)

    if not local.any():
        return []
    labels, n = ndimage.label(local, structure=np.ones((3, 3)))
    clusters = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        clusters.append(np.stack([origin[0] + (xs + i0 + 0.5) * res,
                                   origin[1] + (ys + j0 + 0.5) * res], axis=1))
    return clusters

class AFSMPCNode(Node):
    def __init__(self):
        super().__init__("afs_mpc_controller")
        p = self.declare_parameter
        self.L_f = p("link_front", 1.1059).value
        self.L_r = p("link_rear", 0.985777778).value
        self.r_disc = p("disc_radius", 1.5).value
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
        self.sqp_iters = p("sqp_iters", 2).value
        self.intent_reset_thresh = p("intent_reset_thresh", 1.0).value
        self.stop_on_infeasible = p("stop_on_infeasible", True).value
        self.eps_tol = p("infeasible_slack_tol", 0.03).value
        self.ctrl_rate = p("control_rate", 10.0).value
        self.map_topic = p("map_topic", "/map").value
        self.joy_topic = p("joy_topic", "/right_controller/joy").value
        self.axis_drive = p("axis_drive", 0).value
        self.axis_steer = p("axis_steer", 1).value
        self.invert_drive = p("invert_drive", False).value
        self.invert_steer = p("invert_steer", True).value
        self.deadzone = p("deadzone", 0.06).value
        self.lpf_alpha = p("joy_lpf_alpha", 0.6).value    # light conditioning; MPC du-cost does the smoothing
        self.joy_timeout = p("joy_timeout", 0.5).value
        self.assist_button = p("assist_button", 0).value   # -1 to disable the toggle
        self.assist_default = p("assist_default", True).value  # start with the MPC filter on/off
        self.reset_button = p("reset_button", 1).value 

        self.max_time = 1 / self.ctrl_rate

        # Set collision disc coordinates
        discs = [(1.5 * self.L_f, 0), (0.5 * self.L_f, 0.0),
                    (-0.5 * self.L_r, 0.0), (-1.0 * self.L_r, 0.0)]
        self.model = AFSModel(self.L_f, self.L_r, self.r_disc, discs, self.dt)
        self.mpc = AFSMPC(self.model, H=self.H, gamma_cbf=self.gamma_cbf,
                            margin=self.margin, influence_R=self.influence_R,
                            q_v=self.q_v, q_w=self.q_w, r_v=self.r_v, r_w=self.r_w,
                            w_term=self.w_term, w_slack=self.w_slack,
                            w_slack_lin=self.w_slack_lin,
                            v_max=self.v_max, omega_max=self.omega_max, g_max=self.g_max,
                            sqp_iters=self.sqp_iters,
                            intent_reset_thresh=self.intent_reset_thresh,
                            eps_tol=self.eps_tol)

        # Fo visualization use shorter timestamps in same horizon
        self.viz_sub = 8
        self.viz_model = AFSModel(self.L_f, self.L_r, self.r_disc, discs,
                                  self.dt / self.viz_sub)

        self.unknown_is_obstacle = p("unknown_is_obstacle", True).value
        self.ogm = StaticOGM(True)
        self._map_msg = None
        self.map_frame = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.state = None; self.pose = None; self.gamma = 0.0
        self.v_cmd = 0.0; self.omega_cmd = 0.0
        self.u_prev = np.zeros(2); self._last_joy = None
        self.assist = bool(self.assist_default)
        self._prev_btn = 0
        self._prev_reset_btn = 0
        self._reversing = False


        self.pub = self.create_publisher(Float64MultiArray, "afs/cmd", 10)
        self.pub_reset = self.create_publisher(Empty, "afs/reset", 10)
        self.pub_map = self.create_publisher(OccupancyGrid, "afs/ogm", 1)
        self.pub_safe = self.create_publisher(OccupancyGrid, "afs/safe_region", 1)
        self.pub_intent = self.create_publisher(Path, "afs/intent_path", 1)
        self.pub_plan = self.create_publisher(Path, "afs/planned_path", 1)
        self.pub_poly = self.create_publisher(MarkerArray, "afs/polytopes", 1)
        self.pub_status = self.create_publisher(MarkerArray, "afs/status", 1)
        self.pub_discs = self.create_publisher(MarkerArray, "afs/discs", 10)
        
        map_qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                                reliability=QoSReliabilityPolicy.RELIABLE,
                                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, map_qos)
        self.create_subscription(Odometry, "odom", self.odom_cb, 10)
        self.create_subscription(Float64, "afs/articulation", self.art_cb, 10)
        self.create_subscription(Joy, self.joy_topic, self.joy_cb, 10)

        self.create_timer(1.0 / self.ctrl_rate, self.control_tick)
        #self.create_timer(0.5, self.publish_map)
        #self.create_timer(0.2, self.publish_safe_region)
        self.get_logger().info(
            f"AFS CBF-MPC ready (H={self.H}, dt={self.dt}). "
            f"Waiting for static map on {self.map_topic}. "
            f"assist={'ON' if self.assist else 'OFF (manual)'}")

    def _publish_discs(self, state):
        """Draws each collision disc as a flat circle at its current
        position, sized to its actual radius -- lets you see the safety
        envelope the MPC is actually enforcing while you drive."""
        arr = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        positions = self.model.disc_cords(state)
        for i, p in enumerate(positions):
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "collision_discs"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2.0 * self.r_disc
            m.scale.z = 0.05
            m.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.35)
            arr.markers.append(m)
        self.pub_discs.publish(arr)
        
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
        if 0 <= self.assist_button < len(msg.buttons):
            b = msg.buttons[self.assist_button]
            if b and not self._prev_btn:
                self.assist = not self.assist
                self.get_logger().info(f"MPC assist {'ON' if self.assist else 'OFF (raw passthrough)'}")
            self._prev_btn = b
        if 0 <= self.reset_button < len(msg.buttons):
            rb = msg.buttons[self.reset_button]
            if rb and not self._prev_reset_btn:
                self.pub_reset.publish(Empty())
                self.mpc.x_prev = None
                self.mpc._useq_prev = None
                self.mpc._u_user_prev = None
                self.u_prev = np.zeros(2)
                self.get_logger().info("Reset requested -> published /afs/reset, cleared MPC warm-start")
            self._prev_reset_btn = rb

    def odom_cb(self, msg):
        if self.map_frame is None:
            return
        src = msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, src, Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.05))
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.pose = None
            self.state = None
            return

        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pth = quat_to_yaw(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        t = tf.transform
        yaw_t = quat_to_yaw(t.rotation.z, t.rotation.w)
        c, s = math.cos(yaw_t), math.sin(yaw_t)
        self.pose = (t.translation.x + c * px - s * py,
                    t.translation.y + s * px + c * py,
                    math.atan2(math.sin(yaw_t + pth), math.cos(yaw_t + pth)))
        self._refresh()

    def art_cb(self, msg):
        self.gamma = float(msg.data); self._refresh()

    def _refresh(self):
        if self.pose is not None:
            self.state = np.array([self.pose[0], self.pose[1], self.pose[2], self.gamma])

    def map_cb(self, msg):
        first = not self.ogm.ready()
        self.ogm.load_from_msg(msg)
        self._map_msg = msg
        self.map_frame = msg.header.frame_id 
        if first:
            self.get_logger().info(
                f"Static map received: {msg.info.width}x{msg.info.height} "
                f"@ {msg.info.resolution} m/cell")

    def control_tick(self):
        if self.state is None or not self.ogm.ready(): return
        if self._last_joy is None or \
            (self.get_clock().now() - self._last_joy).nanoseconds * 1e-9 > self.joy_timeout:
            self.v_cmd = 0.0; self.omega_cmd = 0.0
        u_user = np.array([self.v_cmd, self.omega_cmd])
        if abs(self.u_prev[0]) > 0.05:          # deadband: don't flip at a stop
            self._reversing = self.u_prev[0] < 0.0
        # intent = nonlinear rollout of the *held user command* (unaided path)
        intent = [self.state]
        for _ in range(self.H * self.viz_sub):
            g = intent[-1][3]
            om = u_user[1]
            if (g >= self.g_max and om > 0) or (g <= -self.g_max and om < 0):
                om = 0.0
            q = self.viz_model.f(intent[-1], np.array([u_user[0], om]))
            q[3] = float(np.clip(q[3], -self.g_max, self.g_max))
            intent.append(q)
        self._publish_path(self.pub_intent, self._viz_traj(intent))
        self._publish_discs(self.state)
        if not self.assist:
            # Manual / raw passthrough: no CBF-MPC filtering at all, so you
            # can A/B whether the safety filter is actually doing anything.
            u0 = u_user
            self.u_prev = u0
            self.pub.publish(Float64MultiArray(data=[float(u0[0]), float(u0[1])]))
            cx, cy = self.state[0], self.state[1]
            reach = self.v_max * self.H * self.dt + self.influence_R + self.L_f + self.L_r
            clusters = build_clusters(self.ogm.prob(), self.ogm.res, self.ogm.origin,
                                        self.ogm.nx, self.ogm.ny,
                                        (cx - reach, cy - reach), (cx + reach, cy + reach))
            min_h = self.mpc.min_clearance(self.state, clusters)
            self._publish_status(False, min_h < 0.0, bool(abs(u_user[0]) > 1e-3 or abs(u_user[1]) > 1e-3),
                                    {'min_h_now': min_h, 'slack_now': 0.0}, assist_off=True)
            return

        # clusters in a window around the vehicle
        cx, cy = self.state[0], self.state[1]
        reach = self.v_max * self.H * self.dt + self.influence_R + self.L_f + self.L_r
        clusters = build_clusters(self.ogm.prob(), self.ogm.res, self.ogm.origin,
                                    self.ogm.nx, self.ogm.ny,
                                    (cx - reach, cy - reach), (cx + reach, cy + reach))
        t0 = time.perf_counter()                          
        u0, planned, qbar, ok, info = self.mpc.solve(self.state, self.u_prev, u_user, clusters)
        t1 = time.perf_counter()
        solve_time = t1 - t0
        #self.get_logger().info(f"MPC solve took {solve_time*1000:.1f} ms")
        if solve_time > self.max_time:
            self.get_logger().warn(
                f"MPC solve took {solve_time*1000:.1f} ms, exceeding the "
                f"{self.max_time*1000:.1f} ms control period (rate={self.ctrl_rate} Hz)")

        infeasible = bool(info['infeasible'])         # no safe action for the next step
        unsafe_now = info['min_h_now'] < 0.0          # a disc is already inside the keep-out
        driver_cmd = bool(abs(u_user[0]) > 1e-3 or abs(u_user[1]) > 1e-3)
        if infeasible and self.stop_on_infeasible:
            u0 = np.zeros(2)                          # STOP: refuse to enter the unsafe region
            self.mpc.x_prev = None
            self.mpc._useq_prev = None
            
        self.u_prev = u0
        self.pub.publish(Float64MultiArray(data=[float(u0[0]), float(u0[1])]))
        if planned is not None:
            self._publish_path(self.pub_plan, self._viz_traj(planned))
        self._publish_status(infeasible, unsafe_now, driver_cmd, info)
        if self.viz_polytopes and planned is not None:
            self._publish_polytopes(planned, self.mpc._last_relevant_clusters)

    def _publish_path(self, pub, traj):
        path = Path()
        path.header.frame_id = "odom"
        path.header.stamp = self.get_clock().now().to_msg()
        for q in traj:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(q[0])
            ps.pose.position.y = float(q[1])
            ps.pose.position.z = 0.5          # clear of the 0.3 m body cubes
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
            p = self.model.disc_cords(q)[i]
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

    def _rear_traj(self, traj):
        """Predicted rear-axle path: R = F - L_f*e_f - L_r*e_r per horizon state."""
        out = []
        for q in traj:
            th, g = float(q[2]), float(q[3])
            ef = np.array([math.cos(th), math.sin(th)])
            er = np.array([math.cos(th - g), math.sin(th - g)])
            P = np.array([float(q[0]), float(q[1])]) - self.L_f * ef
            out.append(P - self.L_r * er)
        return out

    def _viz_traj(self, traj):
        """Trajectory of whichever end is leading: front axle driving forward,
        rear axle reversing. In teleop the operator cares where the leading end
        is going, and reversing that is the rear."""
        if self._reversing:
            return self._rear_traj(traj)
        return traj

    def _display_anchor(self):
        """Front axle driving forward, rear axle reversing -- whichever end
        leads is where the operator is looking."""
        if self._reversing:
            R = self._rear_traj([self.state])[0]
            return float(R[0]), float(R[1])
        return float(self.state[0]), float(self.state[1])

    def _publish_status(self, infeasible, unsafe_now, driver_cmd, info, assist_off=False):
        # Status indicator above the vehicle:
        #   BLUE   - assist OFF, raw joystick passthrough (manual A/B test mode)
        #   RED    - no safe action exists (QP needed > eps_tol slack); machine stopped.
        #            Driver should change intent (e.g. reverse) to find a feasible action.
        #   ORANGE - a disc is already inside the keep-out (h<0); recovering.
        #   GREEN  - OK / clear.
        x, y = self._display_anchor()

        if assist_off:
            col = (0.2, 0.4, 1.0)
            txt = "ASSIST OFF (manual)  h=%.2f m" % info['min_h_now']
        elif infeasible:
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
        # Echo the received static map back out on afs/ogm so existing RViz
        # configs pointed at it keep working; the map never changes after receipt.
        if self._map_msg is None:
            return
        self._map_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_map.publish(self._map_msg)

    def publish_safe_region(self):
        # Safe region = free space minus obstacles inflated by (r_disc+margin).
        # A disc CENTRE in a '0' cell keeps that disc at least `margin` clear
        # of any obstacle; '100' cells are the keep-out. The vehicle is safe
        # iff all five disc centres lie in '0'. If a corridor's free channel
        # vanishes here, it is too narrow to thread at this margin/radius.
        if not self.ogm.ready():
            return
        prob = self.ogm.prob()
        occ = prob > 0.6
        known = self.ogm.known()
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

def main(args=None):
    rclpy.init(args=args)
    node = AFSMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub.publish(Float64MultiArray(data=[0.0, 0.0]))
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
