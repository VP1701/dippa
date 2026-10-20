"""CBF-MPC on the neural network actuator model (dynamic_afs_model.AFSModel).

State z = [q, r, s] with q = [x_f, y_f, theta_f, gamma], r = [v_f, omega]
the actual velocities and s the command buffers.  One MPC step is n_sub
network steps with the command held (move blocking), so the MPC step is
n_sub * model.dt.  The previous command is already the newest buffer slot,
u_{k-1} = Pi z_k, so the incremental form needs no extra states:

    z_{k+1} = (A_k + B_k Pi) z_k + B_k du_k + c_k

decision  Z = [ z_1..z_H , du_0..du_{H-1} , eps_1..eps_H , sig_1..sig_H ]
          (dim (n_z+4)H)

The articulation limit is soft (slack sig_k): with the delay, gamma over
the first steps is already decided by the commands in the buffer, and the
network does not know the mechanical end stop, so a hard |gamma_k| <=
g_max can be infeasible.

Obstacle rows are those of AFSMPC: they act on z[0:6] = [q, r], so the
stopping term T_stop n^T G r uses the actual velocity instead of the
command.
"""
import math

import numpy as np
from scipy import sparse

from safe_shared_control.afs_mpc import AFSMPC


class DynamicAFSMPC(AFSMPC):
    def __init__(self, model, n_sub, gamma_cbf=2.5, max_iter=1000,
                 eps_abs=1e-3, pri_tol=1e-2, **kwargs):
        super().__init__(model, gamma_cbf=gamma_cbf, **kwargs)
        self.max_iter = int(max_iter)
        self.eps_abs = float(eps_abs)
        self.pri_tol = float(pri_tol)
        self.n_sub = int(n_sub)
        self.dt = self.n_sub * model.dt
        self.alpha = float(np.clip(gamma_cbf * self.dt, 1e-3, 1.0))

        self.nx = model.n_z
        self.nz = (self.nx + 4) * self.H
        m = model.m
        self.u_idx = np.array([5 + m, 5 + 2 * m])     # newest buffer slots
        self.Pi = np.zeros((2, self.nx))
        self.Pi[[0, 1], self.u_idx] = 1.0

        # u_0 is first read by the network gap steps later and reaches the
        # position one step after that; earlier slacks are already decided.
        first = math.ceil((min(model.gaps) + 2) / self.n_sub)
        self.k_check = min(self.H, first + 1)

        self._r0 = None
        self._s0 = None
        self._planned_rates = None

    # --- index helpers into Z ---
    def iq(self, k):
        """Start of z_k, k = 1..H."""
        return self.nx * (k - 1)

    def idu(self, k):
        """Start of du_k, k = 0..H-1."""
        return self.nx * self.H + 2 * k

    def ieps(self, k):
        """eps_k, k = 1..H."""
        return (self.nx + 2) * self.H + (k - 1)

    def isig(self, k):
        """Articulation limit slack sig_k, k = 1..H."""
        return (self.nx + 3) * self.H + (k - 1)

    def set_actuator_state(self, r, s):
        """Measured velocities r and command buffer s for the next solve."""
        self._r0 = np.asarray(r, float)
        self._s0 = np.asarray(s, float)

    def planned_inputs(self):
        """u_k for k = 0..H-1 from the last solution, or None."""
        if self.x_prev is None:
            return None
        return np.array([self.x_prev[self.iq(k + 1) + self.u_idx]
                         for k in range(self.H)])

    def planned_rates(self):
        """Predicted [v_f, omega] for k = 0..H, or None."""
        return self._planned_rates

    def _rollout(self, z0, u_seq):
        """Nonlinear rollout at the network rate, (H * n_sub + 1, n_z)."""
        zs = [z0]
        for u in u_seq:
            for _ in range(self.n_sub):
                zs.append(self.model.f(zs[-1], u))
        return np.array(zs)

    def _linearize_over_horizon(self, z_sub, u_seq):
        """Incremental-form A, B, c (H, n_z, n_z), (H, n_z, 2), (H, n_z).

        The substep linearizations are chained into one affine map per MPC
        step: z' = A_i z + B u + c_i applied n_sub times.
        """
        us = np.repeat(u_seq, self.n_sub, axis=0)
        A_sub, B_sub, c_sub = self.model.linearize(z_sub[:-1], us)

        A = np.empty((self.H, self.nx, self.nx))
        B = np.empty((self.H, self.nx, 2))
        c = np.empty((self.H, self.nx))
        for k in range(self.H):
            A_k, B_k = np.eye(self.nx), np.zeros((self.nx, 2))
            c_k = np.zeros(self.nx)
            for i in range(k * self.n_sub, (k + 1) * self.n_sub):
                A_k = A_sub[i] @ A_k
                B_k = A_sub[i] @ B_k + B_sub
                c_k = A_sub[i] @ c_k + c_sub[i]
            A[k] = A_k + B_k @ self.Pi
            B[k] = B_k
            c[k] = c_k
        return A, B, c

    def _dynamics_rows(self, z0, A, B, c, rows, cols, data, lo, up, r,
                       tol=1e-12):
        """z_{k+1} - A_k z_k - B_k du_k = c_k, nonzero entries only."""
        idx = np.arange(self.nx)
        for k in range(self.H):
            rows.extend(r + idx)
            cols.extend(self.iq(k + 1) + idx)
            data.extend(np.ones(self.nx))

            bi, bj = np.nonzero(np.abs(B[k]) > tol)
            rows.extend(r + bi)
            cols.extend(self.idu(k) + bj)
            data.extend(-B[k][bi, bj])

            if k == 0:
                rhs = A[0] @ z0 + c[0]          # z_0 is measured
            else:
                ai, aj = np.nonzero(np.abs(A[k]) > tol)
                rows.extend(r + ai)
                cols.extend(self.iq(k) + aj)
                data.extend(-A[k][ai, aj])
                rhs = c[k]
            lo.extend(rhs)
            up.extend(rhs)
            r += self.nx
        return r

    def _box_rows(self, u_user, rows, cols, data, lo, up, r):
        """Command limits on u_k (newest slots of z_{k+1}), the soft
        articulation limit |gamma_k| <= g_max + sig_k, eps_k, sig_k >= 0."""
        limits = []                     # (((col, coef), ...), low, high)
        for k in range(self.H):
            i_u = self.iq(k + 1) + self.u_idx
            limits += [(((i_u[0], 1.0),), -self.v_max, self.v_max),
                       (((i_u[1], 1.0),), -self.omega_max, self.omega_max)]
        for k in range(1, self.H + 1):
            i_g, i_s = self.iq(k) + 3, self.isig(k)
            limits += [(((i_g, 1.0), (i_s, -1.0)), -np.inf, self.g_max),
                       (((i_g, 1.0), (i_s, 1.0)), -self.g_max, np.inf),
                       (((self.ieps(k), 1.0),), 0.0, np.inf),
                       (((i_s, 1.0),), 0.0, np.inf)]

        for entries, low, high in limits:
            for col, coef in entries:
                rows.append(r)
                cols.append(col)
                data.append(coef)
            lo.append(low)
            up.append(high)
            r += 1
        return r

    def _build_cost(self, u_user):
        """Tracking of u_user, du smoothness, terminal speed and slacks.

        The terminal cost weights both the actual v_f at z_H and the last
        speed command: with the delay, commands still in the buffer keep
        the machine moving after the horizon.
        """
        Pd = np.full(self.nz, 1e-6)
        qv = np.zeros(self.nz)
        for k in range(self.H):
            i_u = self.iq(k + 1) + self.u_idx
            Pd[i_u] += 2 * self.Qu
            qv[i_u] += -2 * self.Qu * u_user
            i_du = self.idu(k)
            Pd[i_du:i_du + 2] += 2 * self.R

        i_end = self.iq(self.H)
        Pd[i_end + 4] += 2 * self.w_term
        Pd[i_end + self.u_idx[0]] += 2 * self.w_term

        for k in range(1, self.H + 1):
            for i_slack in (self.ieps(k), self.isig(k)):
                Pd[i_slack] += 2 * self.w_slack
                qv[i_slack] += self.w_slack_lin
        return sparse.diags(Pd).tocsc(), qv

    def _assemble_and_solve(self, P, qv, rows, cols, data, lo, up, x_ws):
        """OSQP with an iteration cap for real time.

        Stopped at max_iter, the iterate is still used if its constraint
        violation (primal residual) is below pri_tol; a warm-started ADMM
        iterate is usually close to the solution by then.
        """
        A = sparse.csc_matrix((data, (rows, cols)), shape=(len(lo), self.nz))
        self.prob.setup(P=P, q=qv, A=A, l=np.array(lo), u=np.array(up),
                        verbose=False, warm_starting=True,
                        max_iter=self.max_iter, eps_abs=self.eps_abs,
                        eps_rel=self.eps_abs, polish=True)
        if x_ws is not None and len(x_ws) == self.nz:
            self.prob.warm_start(x=x_ws)
        res = self.prob.solve()
        self.last_osqp = res.info
        # prim_res in OSQP >= 1.0, pri_res before
        pri_res = getattr(res.info, "prim_res",
                          getattr(res.info, "pri_res", np.inf))
        capped = (res.info.status == "maximum iterations reached"
                  and pri_res < self.pri_tol)
        ok = ((res.info.status_val in (1, 2) or capped)
              and res.x is not None and np.all(np.isfinite(res.x)))
        return (res.x if ok else None), ok

    def _info(self, q0, field, x):
        if x is None:
            slack_now = slack_max = float("inf")
        else:
            slack_now = float(x[self.ieps(self.k_check)])
            slack_max = float(max(x[self.ieps(k)]
                                  for k in range(1, self.H + 1)))
        return {'min_h_now': self.min_clearance(q0, field),
                'slack_now': slack_now, 'slack_max': slack_max,
                'infeasible': x is None or slack_now > self.eps_tol,
                'solver_failed': x is None,
                'cover_incomplete': self.cover_incomplete,
                'seed_excluded': self.seed_excluded,
                'out_of_window': field.out_of_window}

    def solve(self, q0, u_prev, u_user, field):
        """Same interface as AFSMPC.solve.

        r and s come from set_actuator_state().  u_prev is not used: the
        previous command is the newest buffer slot, Pi z_0.
        """
        if self._r0 is None:
            raise RuntimeError("set_actuator_state() before solve()")
        q0 = np.asarray(q0, float)
        u_user = np.asarray(u_user, float)
        z0 = np.concatenate([q0, self._r0, self._s0])
        u_last = self.Pi @ z0

        prev = self._u_user_prev
        flip = (prev is None
                or u_user[0] * prev[0] < -0.05
                or np.linalg.norm(u_user - prev) > self.intent_reset_thresh)
        if self._useq_prev is not None and not flip:
            u_seq = np.vstack([self._useq_prev[1:], self._useq_prev[-1:]])
        else:
            u_seq = np.tile(u_user, (self.H, 1))
        self._u_user_prev = u_user.copy()

        P, qv = self._build_cost(u_user)
        x_ws = self.x_prev
        best = None
        for _ in range(self.sqp_iters):
            z_sub = self._rollout(z0, u_seq)
            zbar = z_sub[::self.n_sub]
            A, B, c = self._linearize_over_horizon(z_sub, u_seq)

            rows, cols, data, lo, up = [], [], [], [], []
            r = self._dynamics_rows(z0, A, B, c, rows, cols, data, lo, up, 0)
            r = self._obstacle_constraint_rows(zbar, field, z0[:6], rows,
                                               cols, data, lo, up, r)
            r = self._box_rows(u_user, rows, cols, data, lo, up, r)
            x, ok = self._assemble_and_solve(P, qv, rows, cols, data, lo, up,
                                             x_ws)
            if not ok:
                break
            x_ws = x
            u_seq = np.array([x[self.iq(k + 1) + self.u_idx]
                              for k in range(self.H)])
            best = (x, u_seq, zbar)

        if best is None:
            self._useq_prev = None
            self._planned_rates = None
            return np.zeros(2), None, [q0], False, self._info(q0, field,
                                                              None)

        x, u_seq, zbar = best
        self.x_prev = x
        self._useq_prev = u_seq
        planned = [q0] + [x[self.iq(k):self.iq(k) + 4]
                          for k in range(1, self.H + 1)]
        self._planned_rates = np.array(
            [self._r0] + [x[self.iq(k) + 4:self.iq(k) + 6]
                          for k in range(1, self.H + 1)])
        u0 = u_last + x[self.idu(0):self.idu(0) + 2]
        return u0, planned, list(zbar[:, :4]), True, self._info(q0, field, x)