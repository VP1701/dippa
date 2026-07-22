# afs_mpc.py
import math
import numpy as np
from scipy import ndimage, sparse
import osqp 
import time


class AFSMPC:
    def __init__(self, model, H=8, gamma_cbf=2.5, margin=0.1,
                 influence_R=1.5, q_v=1.0, q_w=0.3, r_v=0.1, r_w=0.25,
                 w_term=1.0, w_slack=1e4, w_slack_lin=1e3,
                 v_max=1.2, omega_max=1.2, g_max=0.75,
                 sqp_iters=1, intent_reset_thresh=1.0, eps_tol=0.03):
        self.model = model
        self.H = H
        self.dt =  model.dt
          
        self.alpha = float(np.clip(gamma_cbf * self.dt, 1e-3, 1.0))
        self.margin = margin
        self.influence_R = influence_R
        self.Qu = np.array([q_v, q_w]); self.R = np.array([r_v, r_w])
        self.w_term, self.w_slack = w_term, w_slack
        self.w_slack_lin = w_slack_lin
        self.v_max = v_max
        self.omega_max = omega_max
        self.g_max = g_max
        self.eps_tol = float(eps_tol)     # first-step slack above this => "no safe action"
        self.nz = 9 * H
        self.sqp_iters = int(sqp_iters)
        self.intent_reset_thresh = float(intent_reset_thresh)
        self.x_prev = None
        self._useq_prev = None
        self._u_user_prev = None
        self._last_relevant_clusters = []

    # --- index helpers into Z ---
    def iq(self, k):  return 6 * (k - 1)            # q~_k start (k=1..H)
    def idu(self, k): return 6 * self.H + 2 * k     # du_k start (k=0..H-1)
    def ieps(self, k): return 8 * self.H + (k - 1)  # eps_k      (k=1..H)

    def _linearize_over_horizon(self, qbar, u_seq):
        """ Linearizes the LTV system per timestep
        
        """

        A_incremental_list = []
        B_incremental_list = []
        c_incremental_list = []

        for k in range(self.H):
            # Get linearized matrices
            A, B, c = self.model.linearize(qbar[k], u_seq[k])

            # Construct augmented form for incremental control
            A_aug = np.block([[A, B], [np.zeros((2,4)), np.eye(2)]])
            B_aug = np.vstack([B, np.eye(2)])
            c_aug = np.concatenate([c, np.zeros(2)])
            A_incremental_list.append(A_aug)
            B_incremental_list.append(B_aug)
            c_incremental_list.append(c_aug)

        return A_incremental_list, B_incremental_list, c_incremental_list

    def _nominal_trajectory(self, q0, u_seq):
        """ Calculates the nominal trajectory for the vehicle
        based on the current state and given control sequence
        
        Args:
            - q0: current state
            - u_seq: control sequence
            
        Return:
            - qbar: list of predicted future states
        """

        qbar = [q0]
        for k in range(self.H):
            q_pred = self.model.f(qbar[k], u_seq[k])
            qbar.append(q_pred)


        return qbar

    def _disc_cluster_distance(self, disc_position, cluster):
        """ Calculates the dinstace from disc position to the nearest
        obastacle point in cluster of obstacle points"""

        # Calculate euclidian distance to each cluster point
        dist = np.sqrt(((cluster - disc_position[None, :])**2).sum(axis=1))

        index = dist.argmin()
        smallest_dist = dist[index]

        if smallest_dist > self.influence_R:
            return None
        
        nearest_point = cluster[index]
        h = smallest_dist - self.model.r_disc - self.margin
        return nearest_point, smallest_dist, h

    def _disc_cluster_horizon(self, disc_positions, cluster):
        """ Calculates the nearest point, smallest distance and h for 
        one obstacle disc and one obstacle cluster per timestep in horizon. 
        If oobstacle not in range for that timestep result is None"""
        diff = disc_positions[:, None, :] - cluster[None, :, :]   # (H+1, N, 2)
        d2 = (diff ** 2).sum(-1)                                   # (H+1, N)
        idx = d2.argmin(axis=1)                                    # (H+1,)
        dist = np.sqrt(d2[np.arange(len(disc_positions)), idx])    # (H+1,)
        nearest = cluster[idx]                                     # (H+1, 2)
        h = dist - self.model.r_disc - self.margin                       # (H+1,)

        results = []
        for k in range(len(disc_positions)):
            if dist[k] > self.influence_R:
                results.append(None)
            else:
                results.append((nearest[k], dist[k], h[k]))
        return results

    def _linearize_obstacle_clearance(self, disc_position, qbar_k, nearest_point, dist, h, disc_jacobian_k):
        """ Linearizes the obstacle collision circle to a tangent line which creates
        a half plane shaped safe region
        
        Args:
        
        
        Returns;
        
        """

        # calculate normal vector between obstacle point and vehicle point
        normal = (disc_position - nearest_point) / dist
        a = normal @ disc_jacobian_k
        b = h - a @ qbar_k
        return a, b

    def _linearize_disc_cluster_horizon(self, disc_positions, qbar, disc_jacobians, results):
        """
        
        
        """
        
        linearized = []

        for k, result in enumerate(results):
            if result is None:
                linearized.append(None)
                continue

            nearest_point, dist, h = result

            a, b = self._linearize_obstacle_clearance(disc_positions[k], qbar[k], nearest_point, dist, h,
                disc_jacobians[k])

            linearized.append((a, b))

        return linearized

    def _obstacle_rows(self, linearized, q0, rows, cols, data, lo, up, r):
        
        for k in range(self.H):
            if linearized[k] is None or linearized[k + 1] is None:
                continue
            
            a_k, b_k = linearized[k]
            a_k_next, b_k_next = linearized[k + 1]

            iq_k_next = self.iq(k + 1)

            for b in range(4):
                rows.append(r)
                cols.append(iq_k_next + b)
                data.append(a_k_next[b])

            rows.append(r)
            cols.append(self.ieps(k + 1))
            data.append(1.0)

            if k == 0:
                lo.append((1 - self.alpha)*(a_k @ q0 + b_k) - b_k_next)
            else:
                iq_k = self.iq(k)
                for b in range(4):
                    rows.append(r) #
                    cols.append(iq_k + b)
                    data.append(-(1 - self.alpha) * a_k[b])
                lo.append((1 - self.alpha)*b_k - b_k_next)
            
            up.append(np.inf)
            r += 1

        return r

    def _obstacle_constraint_rows(self, qbar, clusters, q0, rows, cols,
                                  data, lo, up, r):
        """ Appends obstacle constraint rows for every collsiion disc
        agains every obstacle cluster
        
        
        
        """     

        num_steps = self.H + 1
        num_discs = len(self.model.discs)
        
        # Get position and jacobian for each disc over the whole horizon
        position_at_step = []
        jacobian_at_step = [] 
        for k in range(num_steps):
            position_at_step.append(self.model.disc_cords(qbar[k]))
            jacobian_at_step.append(self.model.disc_jacobians(qbar[k])) 
        #print(f"Len clusters before: {len(clusters)}")
        # Filter out irrelevant clusters
        all_positions = np.array(position_at_step).reshape(num_steps * num_discs, 2)
        relevant_clusters = []
        for cluster in clusters:
            diff = all_positions[:, None, :] - cluster[None, :, :]
            d2 = (diff ** 2).sum(-1)
            if d2.min() <= self.influence_R ** 2:
                relevant_clusters.append(cluster)
        clusters = relevant_clusters
        self._last_relevant_clusters = relevant_clusters
        #print(f"Len clusters after: {len(clusters)}")
        # Get position and jacobian for a disc over the whole horizon
        for disc_index in range(num_discs):
            disc_positions = []
            disc_jacobians = []

            for k in range(num_steps):
                disc_positions.append(position_at_step[k][disc_index])
                disc_jacobians.append(jacobian_at_step[k][disc_index])
            disc_positions = np.array(disc_positions)

            # build obstacle row for disc against cluster
            for cluster in clusters:
                horizon = self._disc_cluster_horizon(disc_positions, cluster)

                linearized_obstacle = self._linearize_disc_cluster_horizon(
                disc_positions, qbar, disc_jacobians, horizon)

                r = self._obstacle_rows(linearized_obstacle, q0, rows, cols, data,
                                        lo, up, r)

        return r


    def _dynamics_rows(self, q0, u_prev, A_list, B_list, c_list, rows, cols,
                    data, lo, up, r):
        q_aug_0 = np.concatenate([q0, u_prev])

        for k in range(self.H):
            A_k = A_list[k]
            B_k = B_list[k]
            c_k = c_list[k]

            iq_k_next = self.iq(k + 1)
            idu_k = self.idu(k)

            # q~_{k+1} block: identity coefficients, one row per component.
            for row in range(6):
                rows.append(r + row)
                cols.append(iq_k_next + row)
                data.append(1.0)

            # -B_k @ du_k block: each of the 6 rows gets 2 column entries.
            for row in range(6):
                for col in range(2):
                    rows.append(r + row)
                    cols.append(idu_k + col)
                    data.append(-B_k[row, col])

            if k == 0:
                # q~_0 is known, not a variable: fold A_k @ q~_0 into rhs.
                rhs = A_k @ q_aug_0 + c_k
            else:
                # q~_k block: -A_k coefficients, 6x6 entries.
                iq_k = self.iq(k)
                for row in range(6):
                    for col in range(6):
                        rows.append(r + row)
                        cols.append(iq_k + col)
                        data.append(-A_k[row, col])
                rhs = c_k

            # Equality: lo == up == rhs, one value per row.
            for row in range(6):
                lo.append(rhs[row])
                up.append(rhs[row])

            r += 6

        return r


    def _box_rows(self, u_user, rows, cols, data, lo, up, r):
        """Appends actuator/state-limit and slack-nonnegativity rows.

        - |v| <= v_max, |omega| <= omega_max, for u_k (k = 0..H-1)
        - |gamma| <= g_max, for q_k's gamma component (k = 1..H)
        - eps_k >= 0 (k = 1..H)

        Args:
            rows, cols, data, lo, up: shared sparse-triplet lists to
                append to (mutated in place).
            r: int, next free row index.

        Returns:
            r: updated next free row index.
        """
        """omega_deadzone = 1e-3  
        drive_sign = 1.0 if u_user[0] >= 0.0 else -1.0
        curv_intent = drive_sign * u_user[1]      # sign of intended path curvature

        if curv_intent > omega_deadzone:
            omega_lo, omega_hi = 0.0, self.omega_max
        elif curv_intent < -omega_deadzone:
            omega_lo, omega_hi = -self.omega_max, 0.0
        else:
            omega_lo, omega_hi = -0.1 * self.omega_max, 0.1 * self.omega_max"""

        # Input limits: u_k lives in q~_{k+1}'s trailing 2 components.
        for k in range(self.H):
            iu = self.iq(k + 1) + 4
            rows.append(r); cols.append(iu); data.append(1.0)
            lo.append(-self.v_max); up.append(self.v_max)
            r += 1

            rows.append(r); cols.append(iu + 1); data.append(1.0)
            lo.append(-self.omega_max); up.append(self.omega_max)
            r += 1

        # Articulation limit: gamma is q_k's 4th component.
        for k in range(1, self.H + 1):
            i_gamma = self.iq(k) + 3
            rows.append(r); cols.append(i_gamma); data.append(1.0)
            lo.append(-self.g_max); up.append(self.g_max)
            r += 1

        # Slack non-negativity.
        for k in range(1, self.H + 1):
            rows.append(r); cols.append(self.ieps(k)); data.append(1.0)
            lo.append(0.0); up.append(np.inf)
            r += 1

        return r


    def min_clearance(self, q, clusters):
        """Smallest disc clearance across all discs and obstacle clusters,
        at a single pose q (h < 0 means a disc is already in the keep-out).

        Args:
            q: array (4,), a single state (not a horizon).
            clusters: list of obstacle point clusters, each an (N, 2) array.

        Returns:
            hmin: float, the smallest clearance found (np.inf if no cluster
            is within influence range of any disc).
        """
        disc_positions = self.model.disc_cords(q)
        hmin = np.inf
        for disc_position in disc_positions:
            for cluster in clusters:
                result = self._disc_cluster_distance(disc_position, cluster)
                if result is None:
                    continue
                _, _, h = result
                hmin = min(hmin, h)
        return float(hmin)


    def _build_cost(self, u_user):
        """Builds the diagonal quadratic cost P and linear cost qv for the QP.

        Args:
            u_user: array (2,), the driver's held command [v, omega].

        Returns:
            P: sparse (nz, nz) diagonal cost matrix.
            qv: array (nz,), linear cost vector.
        """
        Pd = np.full(self.nz, 1e-6)   # tiny regularization on every variable
        qv = np.zeros(self.nz)

        # Tracking cost: penalize u_k (in q~_{k+1}) deviating from u_user.
        for k in range(self.H):
            iu = self.iq(k + 1) + 4
            Pd[iu] += 2 * self.Qu[0]
            qv[iu] += -2 * self.Qu[0] * u_user[0]
            Pd[iu + 1] += 2 * self.Qu[1]
            qv[iu + 1] += -2 * self.Qu[1] * u_user[1]

        # Smoothness cost: penalize du_k.
        for k in range(self.H):
            idu = self.idu(k)
            Pd[idu] += 2 * self.R[0]
            Pd[idu + 1] += 2 * self.R[1]

        # Terminal cost: soft penalty on terminal speed v_f.
        Pd[self.iq(self.H) + 4] += 2 * self.w_term

        # Slack cost: quadratic + linear exact-penalty term.
        for k in range(1, self.H + 1):
            Pd[self.ieps(k)] += 2 * self.w_slack
            qv[self.ieps(k)] += self.w_slack_lin

        P = sparse.diags(Pd).tocsc()
        return P, qv


    def _assemble_and_solve(self, P, qv, rows, cols, data, lo, up, x_ws):
        """Builds the sparse constraint matrix, solves the QP, and returns
        the solution.

        Args:
            P: sparse (nz, nz) diagonal cost matrix, from _build_cost.
            qv: array (nz,), linear cost vector, from _build_cost.
            rows, cols, data: sparse-triplet entries for the constraint
                matrix Ac, accumulated by the row-builder methods.
            lo, up: lists of lower/upper bounds, one per constraint row.
            x_ws: array (nz,) or None, warm-start guess for the primal
                solution.

        Returns:
            x: array (nz,), the solution, or None if the solve failed.
            ok: bool, whether the solve succeeded.
        """
        num_rows = len(lo)
        Ac = sparse.csc_matrix((data, (rows, cols)), shape=(num_rows, self.nz))

        prob = osqp.OSQP()
        prob.setup(P=P, q=qv, A=Ac, l=np.array(lo), u=np.array(up),
                verbose=False, warm_starting=True, max_iter=8000,
                eps_abs=1e-4, eps_rel=1e-4, polish=True)

        if x_ws is not None and len(x_ws) == self.nz:
            try:
                prob.warm_start(x=x_ws)
            except Exception:
                pass

        res = prob.solve()
        ok = (res.info.status_val in (1, 2)
            and res.x is not None
            and np.all(np.isfinite(res.x)))
        return (res.x if ok else None), ok

    def solve(self, q0, u_prev, u_user, clusters):
        """Solves the CBF-MPC QP for one control tick.

        Args:
            q0: array (4,), current state.
            u_prev: array (2,), previously applied input.
            u_user: array (2,), driver's held command [v, omega].
            clusters: list of obstacle point clusters, each an (N, 2) array.

        Returns:
            u0: array (2,), the command to apply now.
            planned: list of H+1 states, the MPC's planned trajectory
                (None if the solve failed).
            qbar: list of H+1 states, the last nonlinear nominal used.
            ok: bool, whether a feasible solve was found.
            info: dict with diagnostic fields (min_h_now, slack_now,
                slack_max, infeasible, solver_failed).
        """
        q0 = np.asarray(q0, float)
        u_prev = np.asarray(u_prev, float)
        u_user = np.asarray(u_user, float)

        # Warm-start the nominal input sequence, unless the driver's intent
        # just flipped (forward<->reverse or a large command jump), in which
        # case we reseed from the held command so the new intent takes
        # effect immediately instead of crawling out of the stale plan.
        prev = self._u_user_prev
        flip = (prev is None
                or u_user[0] * prev[0] < -0.05
                or np.linalg.norm(u_user - prev) > self.intent_reset_thresh)
        if self._useq_prev is not None and not flip:
            u_seq = np.vstack([self._useq_prev[1:], self._useq_prev[-1:]])
        else:
            u_seq = np.tile(u_user, (self.H, 1))
        self._u_user_prev = np.array(u_user, float)

        x_ws = self.x_prev
        P, qv = self._build_cost(u_user)
        best = None
        qbar = None

        for _ in range(self.sqp_iters):
            t0 = time.perf_counter()
            qbar = self._nominal_trajectory(q0, u_seq)
            A_list, B_list, c_list = self._linearize_over_horizon(qbar, u_seq)
            t1 = time.perf_counter()

            rows, cols, data, lo, up = [], [], [], [], []
            r = 0
            r = self._dynamics_rows(q0, u_prev, A_list, B_list, c_list, rows, cols, data, lo, up, r)
            t2 = time.perf_counter()
            r = self._obstacle_constraint_rows(qbar, clusters, q0, rows, cols, data, lo, up, r)
            t3 = time.perf_counter()
            r = self._box_rows(u_user, rows, cols, data, lo, up, r)
            t4 = time.perf_counter()

            x, ok = self._assemble_and_solve(P, qv, rows, cols, data, lo, up, x_ws)
            t5 = time.perf_counter()

            #print(f"rollout+lin={1e3*(t1-t0):.1f}ms dyn={1e3*(t2-t1):.1f}ms "f"obstacles={1e3*(t3-t2):.1f}ms box={1e3*(t4-t3):.1f}ms "f"osqp={1e3*(t5-t4):.1f}ms")
            if not ok:
                break

            x_ws = x
            u_seq = np.array([x[self.iq(k + 1) + 4:self.iq(k + 1) + 6]
                            for k in range(self.H)])
            planned = [q0] + [x[self.iq(k):self.iq(k) + 4]
                            for k in range(1, self.H + 1)]
            best = (u_prev + x[self.idu(0):self.idu(0) + 2], planned,
                    list(qbar), x, u_seq)

        if best is None:
            # SAFE FALLBACK: a failed/maxed-out solve must NOT hold the last
            # (possibly forward) command -- that would drive through a wall.
            # Stop instead; the next tick re-solves from rest.
            self._useq_prev = None
            info = {'min_h_now': self.min_clearance(q0, self._last_relevant_clusters),
                    'slack_now': float('inf'), 'slack_max': float('inf'),
                    'infeasible': True, 'solver_failed': True}
            return np.zeros(2), None, [q0], False, info

        u0, planned, qbar, x, u_seq = best
        self.x_prev = x
        self._useq_prev = u_seq
        slack_now = float(x[self.ieps(1)]) if self.H >= 1 else 0.0
        slack_max = float(max(x[self.ieps(k)] for k in range(1, self.H + 1)))
        info = {'min_h_now': self.min_clearance(q0, self._last_relevant_clusters),
                'slack_now': slack_now, 'slack_max': slack_max,
                'infeasible': slack_now > self.eps_tol, 'solver_failed': False}
        return u0, planned, qbar, True, info

