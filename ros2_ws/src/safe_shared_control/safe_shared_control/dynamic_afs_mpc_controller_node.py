#!/usr/bin/env python3
"""
dynamic_afs_mpc_controller_node.py -- CBF-MPC on the neural network model
========================================================================
Same topics, visualization and parameters as afs_mpc_controller_node.py,
but the MPC predicts with the actuator network (mpc_model_01.pt) instead of
assuming the command is the velocity:

    z = [q, r, s],  q = [x_f, y_f, theta_f, gamma],  r = [v_f, omega]
    one MPC step (mpc_dt) = mpc_dt * rate network steps, command held

The node keeps its own copy of the command buffer s at the network rate.
At the start of every tick the command held since the previous update (the
last one published: clipped, zeros on stop, passthrough with assist off)
is pushed once per network sample of elapsed ROS time, so a late tick
still leaves the buffer matching what the machine received.  mpc_dt * rate
must be a whole number.

The network was validated over 2.4 s rollouts, so run with
horizon * mpc_dt <= 2.4, e.g. horizon:=12 with mpc_dt 0.2.

Extra subscriptions: /afs/rates (Float64MultiArray [v_f, omega], actual
                     velocities), /afs/reset (Empty, clears the buffer)
Extra parameters:    mpc_model_path, torch_threads,
                     osqp_max_iter (1000), osqp_eps (1e-3): iteration cap
                     and tolerance; a capped iterate is used if its
                     constraint violation is below 1e-2
"""
import os

import numpy as np
import rclpy
import torch
from std_msgs.msg import Empty, Float64MultiArray

from safe_shared_control.afs_mpc_controller_node import AFSMPCNode
from safe_shared_control.dynamic_afs_model import AFSModel
from safe_shared_control.dynamic_afs_mpc import DynamicAFSMPC
from safe_shared_control.state_plant import build_plant


def model_path(name):
    """models/<name> next to this script, else in the package share."""
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "models", name)
    if os.path.exists(local):
        return local
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("safe_shared_control")
        candidate = os.path.join(share, "models", name)
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    return local  # torch.load raises a clear FileNotFoundError


def whole(value, name):
    """value as an int, or a RuntimeError naming the setting."""
    if value < 1 or abs(value - round(value)) > 1e-9:
        raise RuntimeError(f"{name} = {value:g} must be a whole number >= 1")
    return int(round(value))


class DynamicAFSMPCNode(AFSMPCNode):
    def __init__(self):
        super().__init__()
        p = self.declare_parameter
        path = p("mpc_model_path", model_path("mpc_model_01.pt")).value
        torch.set_num_threads(p("torch_threads", 1).value)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        if cfg.get("input_set", "cmd") != "cmd":
            raise RuntimeError(f"{path} was trained on low-level inputs, the "
                               "MPC needs a velocity command model")
        rate = float(cfg["rate"])
        n_sub = whole(self.dt * rate, "mpc_dt * model rate")
        self.dt_net = 1.0 / rate

        # The scales are the largest |value| in the training data; the
        # network has not seen anything faster.
        for name, limit, trained in (
                ("v_max", self.v_max, ckpt["scales"]["v"]),
                ("omega_max", self.omega_max, ckpt["scales"]["w"])):
            if limit > trained + 1e-6:
                self.get_logger().warn(
                    f"{name} = {limit:g} is above the training data maximum "
                    f"{trained:.3g}: the network extrapolates there")

        net = build_plant(cfg)
        net.load_state_dict(ckpt["model"])
        self.dyn_model = AFSModel(self.L_f, self.L_r, self.r_disc,
                                  self.model.discs, 1.0 / rate, net,
                                  ckpt["scales"], cfg["inputs"],
                                  cfg["gaps"])
        self.mpc = DynamicAFSMPC(
            self.dyn_model, n_sub,
            max_iter=p("osqp_max_iter", 1000).value,
            eps_abs=p("osqp_eps", 1e-3).value, **self._mpc_kwargs())

        self.r = None
        self.s = np.zeros(2 * self.dyn_model.m)
        self._t_buffer = None           # time the buffer is up to [s]
        self.u_lim = np.array([self.v_max, self.omega_max])
        self.create_subscription(Float64MultiArray, "afs/rates",
                                 self.rates_cb, 10)
        self.create_subscription(Empty, "afs/reset", self.buffer_reset_cb,
                                 10)
        self.get_logger().info(
            f"Dynamic MPC: {path} ({cfg.get('arch', 'mlp')}, {rate:g} Hz, "
            f"gaps {tuple(cfg['gaps'])}, {cfg.get('activation', 'relu')}), "
            f"{n_sub} network steps per "
            f"MPC step, horizon {self.H * self.dt:.1f} s, n_z "
            f"{self.dyn_model.n_z}, slack checked at step "
            f"{self.mpc.k_check}.")

    def rates_cb(self, msg):
        self.r = np.array(msg.data[:2], float)

    def buffer_reset_cb(self, msg):
        self.s[:] = 0.0

    def _advance_buffer(self):
        """Push the command held since the last update, once per network
        sample of elapsed time.

        u_prev is the last command published, which the machine holds
        until the next one.  The reference time moves in whole samples, so
        rounding does not accumulate.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._t_buffer is None:
            self._t_buffer = now
            return
        n = int((now - self._t_buffer) / self.dt_net + 0.5)
        self._t_buffer += n * self.dt_net
        u = np.clip(self.u_prev, -self.u_lim, self.u_lim)
        m = self.dyn_model.m
        for _ in range(min(n, m)):
            self.s = np.concatenate([self.s[1:m], u[:1],
                                     self.s[m + 1:], u[1:]])

    def control_tick(self):
        if self.r is None:
            self.get_logger().warn("waiting for afs/rates (dynamic sim?)",
                                   throttle_duration_sec=5.0)
            return
        self._advance_buffer()
        self.mpc.set_actuator_state(self.r, self.s)
        super().control_tick()

    def _stop_velocities(self):
        return self.mpc.planned_rates()


def main(args=None):
    rclpy.init(args=args)
    node = DynamicAFSMPCNode()
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