# afs_model.py
import math
import numpy as np


class AFSModel:
    def __init__(self, L_f, L_r, r_disc, discs, dt):
        self.L_f = L_f
        self.L_r = L_r
        self.r_disc = r_disc
        self.dt =  dt
        # (kind, offset) for each disc: 3 along front body, 2 along rear body
        self.discs = discs
        self.A_k = np.eye(4)
        self.B_k = np.zeros((4,2))
        self.B_k[3,1] = dt

        self.I = np.eye(4)

    def f_c(self, q, u):
        """ Continuous time kinematics of and AFS-vehicle"""
        _, _, th, g = q
        vf, om = u

        Dq = self.L_f * math.cos(g) + self.L_r

        xfd = vf * math.cos(th)
        yfd = vf * math.sin(th)
        thd = (self.L_r * om + vf * math.sin(g)) / Dq
        gd = om
        return np.array([xfd, yfd, thd, gd])

    def f_euler(self, q, u):
        """Nonlinear one-step (no wrap/clip; limits handled by QP constraints)."""
        xf, yf, th, g = q
        vf, om = u
        Dq = self.L_f * math.cos(g) + self.L_r
        thd = (self.L_r * om + vf * math.sin(g)) / Dq
 
        xf += self.dt * vf * math.cos(th)
        yf += self.dt * vf * math.sin(th)
        th += self.dt * thd
        g += self.dt * om
        return np.array([xf, yf, th, g])

    def f(self, q, u):
        """One RK4 step of the continuous dynamics."""
        k1 = self.f_c(q,u)
        k2 = self.f_c(q + k1*self.dt/2,u)
        k3 = self.f_c(q + k2*self.dt/2,u)
        k4 = self.f_c(q + k3*self.dt,u)

        return q + self.dt/6*(k1 + 2 * k2 + 2 * k3 + k4)
        

    def jac_c(self, q, u):
        _, _, th, g = q
        v_f, om = u

        term1 = self.L_f * np.cos(g) + self.L_r
        sin_theta_f = np.sin(th)
        cos_theta_f = np.cos(th)
        sin_gamma = np.sin(g)
        # calculating A
        A = np.zeros((4, 4))
        A[0,2] = -v_f * sin_theta_f
        A[1,2] = v_f * cos_theta_f
        num = v_f * self.L_f + self.L_f * self.L_r * om * sin_gamma + self.L_r * v_f * np.cos(g)
        den = (term1) ** 2
        A[2,3] = num / den

        # calculating B
        B = np.zeros((4, 2))
        B[0,0] = cos_theta_f
        B[1,0] = sin_theta_f
        B[2,0] = sin_gamma / (term1)
        B[2,1] = self.L_r / (term1)
        B[3, 1] = 1.0

        return A, B


    def linearize_euler(self, q, u, eps=1e-6):
        v_f = u[0]
        omega = u[1]

        theta_f = q[2]
        gamma = q[3]

        term1 = self.L_f * np.cos(gamma) + self.L_r
        sin_theta_f = np.sin(theta_f)
        cos_theta_f = np.cos(theta_f)
        sin_gamma = np.sin(gamma)
        # calculating A_k
        self.A_k[0,2] = -self.dt * v_f * sin_theta_f
        self.A_k[1,2] = self.dt * v_f * cos_theta_f
        num = v_f * self.L_f + self.L_f * self.L_r * omega * sin_gamma + self.L_r * v_f * np.cos(gamma)
        den = (term1) ** 2
        self.A_k[2,3] = self.dt * num / den

        # calculating B_k
        self.B_k[0,0] = self.dt * cos_theta_f
        self.B_k[1,0] = self.dt * sin_theta_f
        self.B_k[2,0] = self.dt * sin_gamma / (term1)
        self.B_k[2,1] = self.dt * self.L_r / (term1)

        # calculate c_k
        F = self.f(q,u)

        c_k = F - self.A_k @ q - self.B_k @ u

        return self.A_k, self.B_k, c_k 

    def linearize(self, q, u, eps=1e-6):
        """A_k, B_k, c_k for the RK4 step: chain rule through the four stages."""
        k1 = self.f_c(q,u)
        q2 = q + k1*self.dt/2
        k2 = self.f_c(q2,u)
        q3 = q + k2*self.dt/2
        k3 = self.f_c(q3,u)
        q4 = q + k3*self.dt
        k4 = self.f_c(q4,u)

        A1, B1 = self.jac_c(q, u)
        A2, B2 = self.jac_c(q2, u)
        A3, B3 = self.jac_c(q3, u)
        A4, B4 = self.jac_c(q4, u)

        dk1_dq = A1
        dk2_dq = A2 @ (self.I + 0.5 * self.dt * dk1_dq)
        dk3_dq = A3 @ (self.I + 0.5 * self.dt * dk2_dq)
        dk4_dq = A4 @ (self.I + self.dt * dk3_dq)

        dk1_du = B1
        dk2_du = B2 + A2 @ (0.5 * self.dt * dk1_du)
        dk3_du = B3 + A3 @ (0.5 * self.dt * dk2_du)
        dk4_du = B4 + A4 @ (self.dt * dk3_du)

        A_k = self.I + (self.dt / 6.0) * (dk1_dq + 2 * dk2_dq + 2 * dk3_dq + dk4_dq)
        B_k = (self.dt / 6.0) * (dk1_du + 2 * dk2_du + 2 * dk3_du + dk4_du)
        c_k = (q + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) - A_k @ q - B_k @ u
        return A_k, B_k, c_k

    def disc_cords(self, q):
        """ Calculates coorinates for the collision discs relative
        to the center link.

        Args:
            - q: State vector of machine (Numpy array)

        Returns:
            - disc_coordinates: list of collision disc coordinates
        """
        xf, yf, th, g = q
        thr = th - g # backbody heading
        front_axle = np.array([xf, yf])


        # x direction vectors
        ef = np.array([np.cos(th), np.sin(th)])
        er = np.array([np.cos(thr), np.sin(thr)])

        # y direction vectors
        ef_perpendicular = np.array([-np.sin(th), np.cos(th)])
        er_perpendicular = np.array([-np.sin(thr), np.cos(thr)])


        # Front axel is measured. Claculate center link position
        center_link = front_axle - self.L_f * ef

        disc_coordinates = []

        # Disc are positioned relative to the center link frame
        for disc_x, disc_y in self.discs:
            if disc_x >= 0.0:
                disc_cord = center_link + disc_x * ef + disc_y * ef_perpendicular
            else:
                disc_cord = center_link + disc_x * er + disc_y * er_perpendicular

            disc_coordinates.append(disc_cord)

        return disc_coordinates


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
        for disc_x, disc_y in self.discs:
            if disc_x >= 0.0:
                J = np.column_stack([
                    [1.0, 0.0],
                    [0.0, 1.0],
                    -self.L_f * ef_perpendicular + disc_x * ef_perpendicular + disc_y * -ef,
                    [0.0, 0.0]
                ])
            else:
                # Chain rule for derivatin sin(th - g) and cos(th - g)
                J = np.column_stack([
                    [1.0, 0.0],
                    [0.0, 1.0],
                    -self.L_f * ef_perpendicular + disc_x * er_perpendicular - disc_y * er,
                    -disc_x * er_perpendicular + disc_y * er
                ])
            Jacobians.append(J)
        return Jacobians
