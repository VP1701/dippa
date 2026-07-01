#!/usr/bin/env python3
"""
afs_controller_node.py  --  ROS2 (Humble) Ponsse-handle controller + CBF filter
===============================================================================
Controller side of the loop. It:
  1. reads the RIGHT Ponsse handle (sensor_msgs/Joy) -> raw command [v_f, omega]
  2. builds an occupancy grid from /scan and extracts per-body CBF constraints
  3. solves the minimal-intervention (v_f, omega) QP
  4. publishes the filtered command on /afs/cmd

Pairs with afs_sim_node.py (front-referenced model):

    THIS --(/afs/cmd)--> afs_sim_node --(/scan,/odom,/afs/articulation)--> THIS

------------------------------------------------------------------- INPUT
Right handle joystick (2 axes): one axis drives forward/back (-> v_f), the other
turns left/right (-> omega). Axis indices, inversions and a deadzone are params,
because the Ponsse handle is not a standard gamepad -- verify with
`ros2 topic echo /right_controller/joy` while moving the stick, then set:
    axis_drive, axis_steer, invert_drive, invert_steer
A button toggles the CBF assist (set `assist_button`, -1 to disable).

------------------------------------------------------------------ INTERFACE
Subscribes: /right_controller/joy (sensor_msgs/Joy), /scan (LaserScan),
            /odom (Odometry), /afs/articulation (Float64)
Publishes:  /afs/cmd (Float64MultiArray [v_f, omega]),
            /afs/ogm (nav_msgs/OccupancyGrid, the built map for RViz)
"""
import math
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import LaserScan, Joy
from nav_msgs.msg import Odometry, OccupancyGrid


# --------------------------------------------------------------------------- #
# Occupancy grid built from observed /scan                                    #
# --------------------------------------------------------------------------- #
class LocalOGM:
    def __init__(self, width, height, res, origin=(0.0, 0.0)):
        self.res = res
        self.origin = np.array(origin, float)
        self.nx, self.ny = int(round(width / res)), int(round(height / res))
        self.logodds = np.zeros((self.ny, self.nx))
        self.L_OCC, self.L_FREE, self.L_CLAMP = 0.85, -0.40, 6.0

    def world_to_cell(self, p):
        c = ((np.asarray(p) - self.origin) / self.res).astype(int)
        return int(c[0]), int(c[1])

    def prob(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds))

    def update_from_scan(self, sx, sy, syaw, rel_angles, ranges, range_max):
        step = self.res * 0.5
        n_steps = max(1, int(range_max / step))
        abs_ang = syaw + np.asarray(rel_angles)
        dirs = np.stack([np.cos(abs_ang), np.sin(abs_ang)], axis=1)
        s = np.arange(1, n_steps + 1) * step
        pts = np.array([sx, sy])[None, None, :] + s[None, :, None] * dirs[:, None, :]
        ix = ((pts[..., 0] - self.origin[0]) / self.res).astype(np.intp)
        iy = ((pts[..., 1] - self.origin[1]) / self.res).astype(np.intp)
        inb = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        r = np.asarray(ranges, float)
        free_mask = (s[None, :] < (r[:, None] - self.res)) & inb
        hit = r < (range_max - 1e-3)
        hit_step = np.clip((r / step).astype(np.intp) - 1, 0, n_steps - 1)
        hit_mask = hit[:, None] & (np.arange(n_steps)[None, :] == hit_step[:, None]) & inb
        np.add.at(self.logodds, (iy[free_mask], ix[free_mask]), self.L_FREE)
        np.add.at(self.logodds, (iy[hit_mask], ix[hit_mask]), self.L_OCC)
        np.clip(self.logodds, -self.L_CLAMP, self.L_CLAMP, out=self.logodds)


# --------------------------------------------------------------------------- #
# Front-referenced vehicle: discs + input jacobians (p_dot = v_f*A + omega*B) #
# --------------------------------------------------------------------------- #
class AFSVehicle:
    def __init__(self, L_f, L_r, r_disc):
        self.L_f, self.L_r, self.r_disc = L_f, L_r, r_disc
        self.front_s = [0.0, L_f / 2, L_f]
        self.rear_t = [L_r / 2, L_r]

    def discs(self, state):
        xf, yf, th_f, g = state
        th_r = th_f - g
        F = np.array([xf, yf])
        ef = np.array([math.cos(th_f), math.sin(th_f)]); nf = np.array([-math.sin(th_f), math.cos(th_f)])
        er = np.array([math.cos(th_r), math.sin(th_r)]); nr = np.array([-math.sin(th_r), math.cos(th_r)])
        Dq = self.L_f * math.cos(g) + self.L_r
        cv = math.sin(g) / Dq
        cw = self.L_r / Dq
        out = []
        for s in self.front_s:                  # p = F - s*ef
            out.append({'p': F - s * ef, 'r': self.r_disc,
                        'A': ef - s * cv * nf, 'B': -s * cw * nf})
        P = F - self.L_f * ef
        for t in self.rear_t:                   # p = P - t*er
            A = ef - self.L_f * cv * nf - t * cv * nr
            B = -self.L_f * cw * nf - t * (cw - 1.0) * nr
            out.append({'p': P - t * er, 'r': self.r_disc, 'A': A, 'B': B})
        return out


def afs_constraints(grid, discs, influence_R, occ_thresh=0.6, max_cons=12):
    occupied = grid.prob() > occ_thresh
    centres = np.array([d['p'] for d in discs])
    i0, j0 = grid.world_to_cell(centres.min(0) - influence_R)
    i1, j1 = grid.world_to_cell(centres.max(0) + influence_R)
    i0, j0 = max(0, i0), max(0, j0)
    i1, j1 = min(grid.nx, i1 + 1), min(grid.ny, j1 + 1)
    local = occupied[j0:j1, i0:i1]
    if not local.any():
        return []
    labels, n = ndimage.label(local)
    clusters = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        clusters.append(np.stack([grid.origin[0] + (xs + i0 + 0.5) * grid.res,
                                   grid.origin[1] + (ys + j0 + 0.5) * grid.res], axis=1))
    cons = []
    for d in discs:
        p = d['p']
        for wpts in clusters:
            dist = np.linalg.norm(wpts - p, axis=1)
            k = int(np.argmin(dist))
            if dist[k] > influence_R:
                continue
            c = wpts[k]; diff = p - c; dd = float(np.linalg.norm(diff))
            if dd < 1e-6:
                continue
            grad = diff / dd
            cons.append({'cv': float(grad @ d['A']), 'cg': float(grad @ d['B']),
                         'h': dd - d['r'], 'c': c})
    cons.sort(key=lambda k: k['h'])
    return cons[:max_cons]


def afs_cbf_qp(z_ref, cons, gamma, v_max, omega_max, g, g_max, k_glim, Rg=0.1):
    z_ref = np.asarray(z_ref, float)
    Q = np.array([1.0, Rg])

    def cost(z):    return float(Q @ (z - z_ref) ** 2)
    def cost_g(z):  return 2.0 * Q * (z - z_ref)

    ineq = []
    for c in cons:
        ineq.append({'type': 'ineq',
                     'fun': (lambda z, c=c: c['cv'] * z[0] + c['cg'] * z[1] + gamma * c['h']),
                     'jac': (lambda z, c=c: np.array([c['cv'], c['cg']]))})
    ineq.append({'type': 'ineq', 'fun': lambda z: k_glim * (g_max - g) - z[1],
                 'jac': lambda z: np.array([0.0, -1.0])})
    ineq.append({'type': 'ineq', 'fun': lambda z: z[1] + k_glim * (g_max + g),
                 'jac': lambda z: np.array([0.0, 1.0])})
    bounds = [(-v_max, v_max), (-omega_max, omega_max)]
    res = minimize(cost, z_ref, jac=cost_g, bounds=bounds, constraints=ineq,
                   method='SLSQP', options={'maxiter': 60, 'ftol': 1e-9})
    return res.x if res.success else z_ref


def quat_to_yaw(qz, qw):
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)


# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class AFSControllerNode(Node):
    def __init__(self):
        super().__init__("afs_controller")
        p = self.declare_parameter
        self.L_f = p("link_front", 0.5).value
        self.L_r = p("link_rear", 0.5).value
        self.r_disc = p("disc_radius", 0.28).value
        self.g_max = p("gamma_max", 0.75).value
        self.v_max = p("v_max", 1.2).value
        self.omega_max = p("omega_max", 1.2).value
        self.gamma_cbf = p("cbf_gamma", 2.0).value
        self.k_glim = p("gamma_limit_gain", 2.0).value
        self.influence_R = p("influence_radius", 1.0).value
        self.ctrl_rate = p("control_rate", 30.0).value
        ow = p("ogm_width", 12.0).value
        oh = p("ogm_height", 8.0).value
        ores = p("ogm_res", 0.06).value
        oorigin = p("ogm_origin", [0.0, 0.0]).value
        # --- joystick mapping (right Ponsse handle) ---
        self.joy_topic = p("joy_topic", "/right_controller/joy").value
        self.axis_drive = p("axis_drive", 0).value       # forward/back axis index
        self.axis_steer = p("axis_steer", 1).value       # turn axis index
        self.invert_drive = p("invert_drive", False).value
        self.invert_steer = p("invert_steer", True).value
        self.deadzone = p("deadzone", 0.06).value
        self.assist_button = p("assist_button", 0).value # -1 to disable
        self.joy_timeout = p("joy_timeout", 0.5).value   # s; zero cmd if joy stale

        self.veh = AFSVehicle(self.L_f, self.L_r, self.r_disc)
        self.ogm = LocalOGM(ow, oh, ores, oorigin)

        self.state = None
        self.pose = None
        self.gamma = 0.0
        self.v_cmd = 0.0
        self.omega_cmd = 0.0
        self.assist = True
        self._prev_btn = 0
        self._last_joy = None

        self.pub = self.create_publisher(Float64MultiArray, "afs/cmd", 10)
        self.pub_map = self.create_publisher(OccupancyGrid, "afs/ogm", 1)
        self.create_subscription(Joy, self.joy_topic, self.joy_cb, 10)
        self.create_subscription(LaserScan, "scan", self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, "odom", self.odom_cb, 10)
        self.create_subscription(Float64, "afs/articulation", self.art_cb, 10)

        self.create_timer(1.0 / self.ctrl_rate, self.control_tick)
        self.create_timer(0.2, self.publish_map)
        self.get_logger().info(
            f"AFS controller ready. Driving from {self.joy_topic} "
            f"(axis_drive={self.axis_drive}, axis_steer={self.axis_steer}). assist=ON")

    # ------------------------------------------------------------------ inputs
    def _dz(self, x):
        if abs(x) < self.deadzone:
            return 0.0
        return (x - math.copysign(self.deadzone, x)) / (1.0 - self.deadzone)

    def joy_cb(self, msg: Joy):
        self._last_joy = self.get_clock().now()
        na = len(msg.axes)
        if na > self.axis_drive and na > self.axis_steer:
            drive = self._dz(msg.axes[self.axis_drive]) * (-1.0 if self.invert_drive else 1.0)
            steer = self._dz(msg.axes[self.axis_steer]) * (-1.0 if self.invert_steer else 1.0)
            self.v_cmd = float(np.clip(drive, -1.0, 1.0) * self.v_max)
            self.omega_cmd = float(np.clip(steer, -1.0, 1.0) * self.omega_max)
        if 0 <= self.assist_button < len(msg.buttons):
            b = msg.buttons[self.assist_button]
            if b and not self._prev_btn:                  # rising edge -> toggle
                self.assist = not self.assist
                self.get_logger().info(f"assist {'ON' if self.assist else 'OFF'}")
            self._prev_btn = b

    def odom_cb(self, msg: Odometry):
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                     quat_to_yaw(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self._refresh_state()

    def art_cb(self, msg: Float64):
        self.gamma = float(msg.data)
        self._refresh_state()

    def _refresh_state(self):
        if self.pose is not None:
            self.state = np.array([self.pose[0], self.pose[1], self.pose[2], self.gamma])

    def scan_cb(self, msg: LaserScan):
        if self.state is None:
            return
        xf, yf, th_f, g = self.state          # FRONT model: LIDAR at the state position
        rel = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.asarray(msg.ranges, float)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        self.ogm.update_from_scan(xf, yf, th_f, rel, ranges, msg.range_max)

    # ----------------------------------------------------------------- control
    def control_tick(self):
        if self.state is None:
            return
        # watchdog: stop if the joystick stream went stale
        if self._last_joy is None or \
           (self.get_clock().now() - self._last_joy).nanoseconds * 1e-9 > self.joy_timeout:
            self.v_cmd = 0.0
            self.omega_cmd = 0.0

        z_ref = np.array([self.v_cmd, self.omega_cmd])
        discs = self.veh.discs(self.state)
        cons = afs_constraints(self.ogm, discs, self.influence_R)
        if self.assist:
            z = afs_cbf_qp(z_ref, cons, self.gamma_cbf, self.v_max, self.omega_max,
                           float(self.state[3]), self.g_max, self.k_glim)
        else:
            z = z_ref
        self.pub.publish(Float64MultiArray(data=[float(z[0]), float(z[1])]))

    def publish_map(self):
        prob = self.ogm.prob()
        data = np.full(prob.shape, -1, dtype=np.int8)
        known = np.abs(self.ogm.logodds) > 1e-6
        data[known] = (prob[known] * 100).astype(np.int8)
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.info.resolution = float(self.ogm.res)
        msg.info.width = self.ogm.nx
        msg.info.height = self.ogm.ny
        msg.info.origin.position.x = float(self.ogm.origin[0])
        msg.info.origin.position.y = float(self.ogm.origin[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = data.flatten().tolist()
        self.pub_map.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AFSControllerNode()
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