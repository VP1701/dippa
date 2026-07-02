
# Ros2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs import TwistStamped


# Math and solver imports
import numpy as np
import casadi as ca
from acados_template import AcadosModel




class CbfMpcSolver:

    def __init__(self, L_f, L_r, r_dics, dt, h, gamma_cbf, n_obs_max):


        self.L_f = L_f
        self.L_r = L_r
        self.r_dics = r_dics
        self.dt = dt
        self.h = h
        self.gamma_cbf = gamma_cbf
        self.n_obs_max = n_obs_max


        self.model, self.sym = self._build_model()


    def _build_model(self):

        """
        Builds a kinematic model of eevee and the collision discs with casadi
        """
        x_f = ca.SX.sym("x_f")
        y_f = ca.SX.sym("y_f")
        theta = ca.SX.sym("theta")
        gamma = ca.SX.sym("gamma")

        # Create state vector z
        x = ca.vertcat(x_f, y_f, theta, gamma)

        v_f = ca.SX.sym("v_f")
        omega = ca.SX.sym("omega")

        u = ca.vertcat(v_f, omega)

        u_ref = ca.SX.sym("u_ref", 2)
        obs_cord = ca.SX.sym("obs_cord", 2 * self.n_obs_max)
        obs_cord_active = ca.SX.sym("obs_cord_active", self.n_obs_max)

        p = ca.vertcat(u_ref, obs_cord, obs_cord_active)

        x_dot = v_f * ca.cos(theta)
        y_dot = v_f * ca.sin(theta)
        theta_dot = (self.L_r * omega + v_f * ca.sin(gamma)) / (self.L_f * ca.cos(gamma) + self.L_r)

        kin = ca.vertcat(x_dot, y_dot, theta_dot, omega)
        kin_dot = ca.SX.sym("kin_dot", 4)

        # Create acados model of Eevee
        model = AcadosModel()
        model.name = "afs_kinematics"
        model.x = x 
        model.u = u 
        model.p = p 
        model.xdot = kin_dot
        model.f_expl_expr = kin
        model.f_impl_expr = kin_dot - kin

        # Collision discs
        theta_r = theta - gamma
        front_axle = ca.vertcat(x_f, y_f)
        
        heading_f = ca.vertcat(ca.cos(theta), ca.sin(theta))
        heading_r = ca.vertcat(ca.cos(theta_r), ca.sin(theta_r))
        center_link = front_axle - self.L_f * heading_f
        rear_axle = center_link - self.L_r * heading_r

        disc_f = front_axle
        disc_r = rear_axle

        sym = {
            "x": x, "u": u, "p": p,
            "u_ref": u_ref, "obs_cord": obs_cord, "obs_cord_active": obs_cord_active,
            "discs": (disc_f, disc_r),
        }

        return model, sym



    def _build_ocp(self, ):
        """
        Build the optimal control problem for the cbf mpc
        """

        x = self.sym["x"]
        u = self.sym["u"]
        p = self.sym["p"]

        u_ref = self.sym["u_ref"]
        obs_cord = self.sym["obs_cord"]
        obs_cord_active = self.sym["obs_cord_active"]

        # Initialize the optimal control problem caontainer
        ocp = AcadosOcp()
        ocp.model = self.model
        ocp.dims.N = self.H

        # tracking cost
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.mode.cost_y_expr = u - u_ref
        ocp.mode.cost_y_expr_e = ca.vertcat(x[3]) 
        ocp.cost.yref = np.zeros(2)
        ocp.cost.yref_e = np.zeros(1)
        ocp.cost.W = np.diag([q_v, q_w])
        ocp.cost.W_e = np.diag([W_terminal])

        # bounds
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([-self.v_max, -self.omega_max])
        ocp.constraints.ubu = np.array([self.v_max, self.omega_max])


        
        # CBF constraints







    def
class SafetyFilter(Node):
    """
    Node for not allowing unsafe user or controller commands
    """

    def __init__(self):
        super().__init__('safety_filter')

        self.declare_parameters(
            namespace='',
            parameters=[
                # TODO: complete parameter list when class is otherwise ready. Make parameters to be read from a config
                ("cbf_gamma", 0.5),

            ]
        )

    











def main(args=None):
    rclpy.init(args=args)



if __name__ == '__main__':
    main()
