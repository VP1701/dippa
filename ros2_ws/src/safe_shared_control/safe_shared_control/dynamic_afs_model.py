# afs_model.py
import math

import numpy as np
import torch
from torch.func import jacrev, vmap


class AFSModel:
    """AFS vehicle with a neural network actuator model.

    Augmented state z = [q, r, s]:
        q = [x_f, y_f, theta_f, gamma]
        r = [v_f, omega]                     actual velocities
        s = [s_v, s_w]                       command buffers, oldest first,
                                             m samples each
    Input u = [u_v, u_w] is the command.
    """

    def __init__(self, L_f, L_r, r_disc, discs, dt, nn, scales, n_hist, gaps):
        self.L_f = L_f
        self.L_r = L_r
        self.r_disc = r_disc
        self.dt = dt
        # (body, dx, dy) per disc, offsets from the hinge along each body
        self.discs = discs
        self._is_front = np.array([b == 'f' for b, _, _ in discs])
        self._dx = np.array([dx for _, dx, _ in discs], float)
        self._dy = np.array([dy for _, _, dy in discs], float)

        self.nn = nn.eval().requires_grad_(False).double()
        self.n_hist = n_hist
        self.gaps = gaps
        self.m = max(gaps) + n_hist - 1
        self.n_z = 6 + 2 * self.m

        self.r_scales = torch.tensor([scales["v"], scales["w"]],
                                     dtype=torch.float64)
        cmd_scales = scales.get("cmd", [scales["v"], scales["w"]])
        self.u_scales = torch.tensor(
            cmd_scales, dtype=torch.float64).repeat_interleave(n_hist)
        self.g_scale = scales["gamma"]

        # Network windows: u(k - gap - n_hist + 1) ... u(k - gap)
        m, g_v, g_w = self.m, gaps[0], gaps[1]
        self._win_v = slice(m - g_v - n_hist + 1, m - g_v + 1)
        self._win_w = slice(2 * m - g_w - n_hist + 1, 2 * m - g_w + 1)
        self._r_jac = vmap(jacrev(self.r_step, argnums=(0, 1, 2)))

        # Constant parts of the linearization: buffer shift S and input E
        self.A_template = np.zeros((self.n_z, self.n_z))
        self.A_template[6:6 + m, 6:6 + m] = np.eye(m, k=1)
        self.A_template[6 + m:, 6 + m:] = np.eye(m, k=1)
        self.B_k = np.zeros((self.n_z, 2))
        self.B_k[5 + m, 0] = 1.0        # newest u_v slot
        self.B_k[5 + 2 * m, 1] = 1.0    # newest u_w slot

        self.eye4 = np.eye(4)

    def q_step(self, q, r):
        """Continuous time kinematics q_dot of an AFS vehicle."""
        _, _, th, g = q
        vf, om = r
        Dq = self.L_f * math.cos(g) + self.L_r

        xfd = vf * math.cos(th)
        yfd = vf * math.sin(th)
        thd = (self.L_r * om + vf * math.sin(g)) / Dq
        gd = om
        return np.array([xfd, yfd, thd, gd])

    def r_step(self, q, r, s):
        """r_dot from the network, physical units. Torch tensors in and out."""
        u = torch.cat([s[self._win_v], s[self._win_w]])
        x = torch.cat([r / self.r_scales, q[3:4] / self.g_scale,
                       u / self.u_scales])
        return self.nn(x) * self.r_scales

    def f(self, z, u):
        """One step of the augmented state z with command u.

        RK4 for the kinematics with r held over the step, Euler for r
        as in the network training, and a shift of the command buffers.
        """
        q, r, s = z[:4], z[4:6], z[6:]
        k1 = self.q_step(q, r)
        k2 = self.q_step(q + k1 * self.dt / 2, r)
        k3 = self.q_step(q + k2 * self.dt / 2, r)
        k4 = self.q_step(q + k3 * self.dt, r)
        q_next = q + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        with torch.inference_mode():
            r_dot = self.r_step(torch.from_numpy(q), torch.from_numpy(r),
                                torch.from_numpy(s)).numpy()
        r_next = r + self.dt * r_dot

        m = self.m
        s_next = np.concatenate([s[1:m], u[:1], s[m + 1:], u[1:]])
        return np.concatenate([q_next, r_next, s_next])

    def jac_c(self, q, r):
        """Continuous kinematic Jacobians d q_dot / d q and d q_dot / d r."""
        _, _, th, g = q
        v_f, om = r

        term1 = self.L_f * np.cos(g) + self.L_r
        sin_theta_f = np.sin(th)
        cos_theta_f = np.cos(th)
        sin_gamma = np.sin(g)
        # calculating A
        A = np.zeros((4, 4))
        A[0, 2] = -v_f * sin_theta_f
        A[1, 2] = v_f * cos_theta_f
        num = (v_f * self.L_f + self.L_f * self.L_r * om * sin_gamma
               + self.L_r * v_f * np.cos(g))
        A[2, 3] = num / term1 ** 2

        # calculating B
        B = np.zeros((4, 2))
        B[0, 0] = cos_theta_f
        B[1, 0] = sin_theta_f
        B[2, 0] = sin_gamma / term1
        B[2, 1] = self.L_r / term1
        B[3, 1] = 1.0
        return A, B

    def linearize(self, zs, us):
        """A_k, B_k, c_k of f for a stack of points zs (H, n_z), us (H, 2).

        Returns A (H, n_z, n_z), B (n_z, 2), c (H, n_z). B is the same for
        every step, so one shared array is returned: don't modify it.
        """
        zs = np.atleast_2d(np.asarray(zs, float))
        us = np.atleast_2d(np.asarray(us, float))
        num_steps = len(zs)
        dt = self.dt

        # Network rows, all steps in one batched call
        zt = torch.from_numpy(zs)
        jq, jr, js = self._r_jac(zt[:, :4], zt[:, 4:6], zt[:, 6:])
        A = np.repeat(self.A_template[None], num_steps, axis=0)
        A[:, 4:6, :4] = dt * jq.numpy()
        A[:, 4:6, 4:6] = np.eye(2) + dt * jr.numpy()
        A[:, 4:6, 6:] = dt * js.numpy()

        # Kinematic rows: chain rule through the RK4 stages, r held
        c = np.empty((num_steps, self.n_z))
        for k in range(num_steps):
            q, r = zs[k, :4], zs[k, 4:6]
            k1 = self.q_step(q, r)
            q2 = q + k1 * dt / 2
            k2 = self.q_step(q2, r)
            q3 = q + k2 * dt / 2
            k3 = self.q_step(q3, r)
            q4 = q + k3 * dt

            A1, B1 = self.jac_c(q, r)
            A2, B2 = self.jac_c(q2, r)
            A3, B3 = self.jac_c(q3, r)
            A4, B4 = self.jac_c(q4, r)

            dk1_dq = A1
            dk2_dq = A2 @ (self.eye4 + 0.5 * dt * dk1_dq)
            dk3_dq = A3 @ (self.eye4 + 0.5 * dt * dk2_dq)
            dk4_dq = A4 @ (self.eye4 + dt * dk3_dq)

            dk1_dr = B1
            dk2_dr = B2 + A2 @ (0.5 * dt * dk1_dr)
            dk3_dr = B3 + A3 @ (0.5 * dt * dk2_dr)
            dk4_dr = B4 + A4 @ (dt * dk3_dr)

            A[k, :4, :4] = self.eye4 + dt / 6 * (
                dk1_dq + 2 * dk2_dq + 2 * dk3_dq + dk4_dq)
            A[k, :4, 4:6] = dt / 6 * (dk1_dr + 2 * dk2_dr
                                      + 2 * dk3_dr + dk4_dr)
            c[k] = self.f(zs[k], us[k]) - A[k] @ zs[k] - self.B_k @ us[k]
        return A, self.B_k, c

    def disc_cords(self, q):
        """ Calculates coordinates for the collision discs relative
        to the center link.

        Args:
            - q: State vector of machine (Numpy array)

        Returns:
            - disc_coordinates: list of collision disc coordinates
        """
        xf, yf, th, g = q
        thr = th - g  # backbody heading
        front_axle = np.array([xf, yf])

        # x direction vectors
        ef = np.array([np.cos(th), np.sin(th)])
        er = np.array([np.cos(thr), np.sin(thr)])

        # y direction vectors
        ef_perpendicular = np.array([-np.sin(th), np.cos(th)])
        er_perpendicular = np.array([-np.sin(thr), np.cos(thr)])

        # Front axle is measured. Calculate center link position
        center_link = front_axle - self.L_f * ef

        disc_coordinates = []

        # Disc are positioned relative to the center link frame
        for body, disc_x, disc_y in self.discs:
            if body == 'f':
                disc_cord = (center_link + disc_x * ef
                             + disc_y * ef_perpendicular)
            else:
                disc_cord = (center_link + disc_x * er
                             + disc_y * er_perpendicular)
            disc_coordinates.append(disc_cord)
        return disc_coordinates

    def disc_geometry(self, qs):
        """P (S,D,2), J (S,D,2,4), G (S,D,2,2) for a stack of states qs (S,4).

        Batched replacement for looping disc_cords / disc_jacobians /
        disc_dyn_jacobians over the horizon.
        """
        qs = np.atleast_2d(np.asarray(qs, float))
        th, g = qs[:, 2], qs[:, 3]
        thr = th - g

        ef = np.stack([np.cos(th), np.sin(th)], -1)
        er = np.stack([np.cos(thr), np.sin(thr)], -1)
        efp = np.stack([-np.sin(th), np.cos(th)], -1)
        erp = np.stack([-np.sin(thr), np.cos(thr)], -1)

        f = self._is_front[None, :, None]
        E = np.where(f, ef[:, None, :], er[:, None, :])
        Ep = np.where(f, efp[:, None, :], erp[:, None, :])
        dx, dy = self._dx[None, :, None], self._dy[None, :, None]

        centre = qs[:, :2] - self.L_f * ef
        P = centre[:, None, :] + dx * E + dy * Ep

        S, D = P.shape[0], P.shape[1]
        J = np.zeros((S, D, 2, 4))
        J[:, :, 0, 0] = 1.0
        J[:, :, 1, 1] = 1.0
        J[:, :, :, 2] = -self.L_f * efp[:, None, :] + dx * Ep - dy * E
        J[:, :, :, 3] = np.where(f, 0.0, -dx * Ep + dy * E)

        term1 = self.L_f * np.cos(g) + self.L_r
        B = np.zeros((S, 4, 2))
        B[:, 0, 0] = np.cos(th)
        B[:, 1, 0] = np.sin(th)
        B[:, 2, 0] = np.sin(g) / term1
        B[:, 2, 1] = self.L_r / term1
        B[:, 3, 1] = 1.0

        return P, J, np.einsum('sdij,sjk->sdik', J, B)

    def disc_jacobians(self, q):
        """J_i = d p_i / d q  (2x4) per disc, evaluated at q.

        Args:
            - q: State vector of machine (Numpy array)

        Return:
            - Jacobians: list of collision disc jacobians
        """
        xf, yf, th, g = q
        thr = th - g

        # x direction vectors
        ef = np.array([np.cos(th), np.sin(th)])
        er = np.array([np.cos(thr), np.sin(thr)])

        # y direction vectors
        ef_perpendicular = np.array([-np.sin(th), np.cos(th)])
        er_perpendicular = np.array([-np.sin(thr), np.cos(thr)])

        Jacobians = []
        for body, disc_x, disc_y in self.discs:
            if body == 'f':
                J = np.column_stack([
                    [1.0, 0.0],
                    [0.0, 1.0],
                    (-self.L_f * ef_perpendicular + disc_x * ef_perpendicular
                     - disc_y * ef),
                    [0.0, 0.0]
                ])
            else:
                # Chain rule for deriving sin(th - g) and cos(th - g)
                J = np.column_stack([
                    [1.0, 0.0],
                    [0.0, 1.0],
                    (-self.L_f * ef_perpendicular + disc_x * er_perpendicular
                     - disc_y * er),
                    -disc_x * er_perpendicular + disc_y * er
                ])
            Jacobians.append(J)
        return Jacobians

    def disc_dyn_jacobians(self, q):
        """G_i = J_i * B_c"""
        _, B_c = self.jac_c(q, np.zeros(2))
        dyn_jacobians = []
        for J in self.disc_jacobians(q):
            G_i = J @ B_c
            dyn_jacobians.append(G_i)

        return dyn_jacobians