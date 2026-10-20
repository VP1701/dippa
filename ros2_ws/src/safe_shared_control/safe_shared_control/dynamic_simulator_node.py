#!/usr/bin/env python3
"""
dynamic_simulator_node.py -- AFS simulator with a neural network actuator model
===============================================================================
Same plant, map, moving obstacle and topics as simulator_node.py, but the
command on /afs/cmd no longer sets the velocities directly. It goes into a
command buffer, and a trained network (MLP or TDNN, any checkpoint of the
training scripts) predicts how the actual velocities r = [v_f, omega]
respond, including the machine's delay:

    z = [q, r, s],  q = [x_f, y_f, theta_f, gamma],  r = [v_f, omega]
    z_{k+1} = f(z_k, u_k)          (dynamic_afs_model.AFSModel.f)

The network runs at its own rate, and sim_rate must be a whole multiple of
it.  At the start of every network step the command is sampled and the
network predicts r and the buffer one network step ahead; the pose is
integrated at sim_rate with r held meanwhile, as the model itself does,
and the new r takes effect at the end of the network step.  With the MPC
network as the plant (the launch file's default), prediction and plant
agree exactly apart from this timing.

Extra publisher:  /afs/rates  std_msgs/Float64MultiArray  data = [v_f, omega]
                  (actual velocities; the controller needs them as state)
The old kinematic simulator (simulator_node.py) is unchanged.
"""
import math
import os

import numpy as np
import rclpy
import torch
from std_msgs.msg import Bool, Float64, Float64MultiArray
from nav_msgs.msg import Odometry

from safe_shared_control.dynamic_afs_model import AFSModel
from safe_shared_control.simulator_node import AFSSimNode, yaw_to_quat
from safe_shared_control.state_plant import build_plant


def _default_model_path(name="sim_model.pt"):
    """models/<name> next to this script, else in the package share."""
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, 'models', name)
    if os.path.exists(local):
        return local
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('safe_shared_control')
        cand = os.path.join(share, 'models', name)
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    return local  # torch.load raises a clear FileNotFoundError


class DynamicAFSSimNode(AFSSimNode):
    def __init__(self):
        super().__init__()
        path = self.declare_parameter("model_path",
                                      _default_model_path()).value
        # A tiny network gains nothing from threads, and idle PyTorch
        # threads busy-wait, taking CPU from the controller and RViz.
        torch.set_num_threads(self.declare_parameter("torch_threads",
                                                     1).value)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        ratio = self.sim_rate / cfg["rate"]
        self.substeps = int(round(ratio))
        if self.substeps < 1 or abs(ratio - self.substeps) > 1e-9:
            raise RuntimeError(
                f"{path} runs at {cfg['rate']} Hz; sim_rate "
                f"{self.sim_rate} Hz must be a whole multiple of it")

        net = build_plant(cfg)
        net.load_state_dict(ckpt["model"])
        discs = ([('f', x, 0.0) for x in self.kin.front_s]
                 + [('r', x, 0.0) for x in self.kin.rear_t])
        self.model = AFSModel(self.L_f, self.L_r, self.r_disc, discs,
                              1.0 / cfg["rate"], net, ckpt["scales"],
                              cfg["inputs"], cfg["gaps"])
        self.z = np.concatenate([self.start_pose,
                                 np.zeros(self.model.n_z - 4)])
        self.tick = 0                   # sim ticks since the last reset
        self.next_rs = None             # r and buffer after the step
        self.pub_rates = self.create_publisher(Float64MultiArray,
                                               "afs/rates", 10)
        self.get_logger().info(
            f"Actuator network loaded from {path}: "
            f"{cfg.get('arch', 'mlp')}, {cfg['rate']} Hz "
            f"({self.substeps} sim ticks per network step), "
            f"gaps {tuple(cfg['gaps'])}, history {cfg['inputs']}, "
            f"buffer {self.model.m} samples per channel.")

    def reset_cb(self, msg):
        super().reset_cb(msg)
        self.z = np.concatenate([self.start_pose,
                                 np.zeros(self.model.n_z - 4)])
        self.tick = 0
        self.next_rs = None

    def sim_step(self):
        # Articulation stop: the joint can't move further, whatever the
        # actuator does, so omega is zero before it enters the step.
        if (self.z[3] >= self.g_max and self.z[5] > 0) or \
                (self.z[3] <= -self.g_max and self.z[5] < 0):
            self.z[5] = 0.0
        if self.tick % self.substeps == 0:
            # start of a network step: sample the command, predict r and
            # the buffer at its end
            u = np.array([
                np.clip(self.cmd[0], -self.v_max, self.v_max),
                np.clip(self.cmd[1], -self.omega_max, self.omega_max)])
            self.next_rs = self.model.f(self.z, u)[4:]
        # pose at sim_rate with r held, as the model integrates it
        self.z[:4] = self.kin.integrate(self.z[:4], self.z[4], self.z[5], self.dt)
        self.tick += 1
        if self.tick % self.substeps == 0:
            self.z[4:] = self.next_rs
        
        self.state = self.z[:4].copy()

        v_f, omega = float(self.z[4]), float(self.z[5])
        now = self.get_clock().now().to_msg()
        F, P, R, th_f, th_r = self.kin.frames(self.state)
        g = float(self.state[3])

        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = "odom"
        od.child_frame_id = "base_link"
        od.pose.pose.position.x = float(F[0])
        od.pose.pose.position.y = float(F[1])
        od.pose.pose.orientation = yaw_to_quat(th_f)
        od.twist.twist.linear.x = v_f
        od.twist.twist.angular.z = self.kin.theta_f_dot(v_f, omega, g)
        self.pub_odom.publish(od)
        self.pub_art.publish(Float64(data=g))
        self.pub_rates.publish(Float64MultiArray(data=[v_f, omega]))

        centres = self.kin.disc_centres(self.state)
        hit = any(self.gt.clearance(c) < self.r_disc for c in centres)
        if self.obs is not None:
            self.obs.step(self.dt)
            p_obs = self.obs.position
            lim = self.r_disc + self.obs.radius
            hit = hit or any(np.hypot(*(c - p_obs)) < lim for c in centres)
            v_obs = self.obs.velocity
            self.pub_obs.publish(Float64MultiArray(data=[
                float(p_obs[0]), float(p_obs[1]),
                float(v_obs[0]), float(v_obs[1]), float(self.obs.radius)]))
        self.pub_col.publish(Bool(data=bool(hit)))

        self._tf(now, "odom", "base_link", F[0], F[1], th_f)
        self._tf(now, "base_link", "hinge", -self.L_f, 0.0, 0.0)
        self._tf(now, "hinge", "rear_link", 0.0, 0.0, -g)
        self.publish_body_markers(now, hit)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicAFSSimNode()
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