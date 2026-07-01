#!/usr/bin/env python3
"""
afs_sim_node.py  --  ROS2 (Humble) test simulator for an articulated vehicle
============================================================================
FRONT-REFERENCED kinematic model (state at the front axle):

    state  q = [x_f, y_f, theta_f, gamma]      input  u = [v_f, omega]
    x_f_dot     = v_f cos(theta_f)
    y_f_dot     = v_f sin(theta_f)
    theta_f_dot = (L_r*omega + v_f*sin(gamma)) / (L_f*cos(gamma) + L_r)
    gamma_dot   = omega

The rear body follows: theta_r = theta_f - gamma. Geometry is anchored at the
front axle F: hinge P = F - L_f*e_f, rear axle R = P - L_r*e_r.

This node is the PLANT ONLY (no safety filtering -- that is the controller node).

------------------------------------------------------------------ INTERFACE
Subscribes:  /afs/cmd  std_msgs/Float64MultiArray   data = [v_f, omega]
             /cmd_vel  geometry_msgs/Twist          linear.x = v_f, angular.z = omega
Publishes:   /scan (sensor_msgs/LaserScan, frame front_lidar),
             /odom (nav_msgs/Odometry, odom->base_link at the FRONT axle),
             /afs/articulation (std_msgs/Float64, gamma),
             /afs/collision (std_msgs/Bool),
             /afs/markers (visualization_msgs/MarkerArray)
TF: odom -> base_link(front axle) -> hinge -> rear_link ; base_link -> front_lidar
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64, Bool, Float64MultiArray
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw):
    q = Quaternion(); q.z = math.sin(yaw * 0.5); q.w = math.cos(yaw * 0.5)
    return q


class GroundTruthMap:
    def __init__(self, width, height, res):
        self.width, self.height, self.res = width, height, res
        self.nx, self.ny = int(round(width / res)), int(round(height / res))
        self.grid = np.zeros((self.ny, self.nx), dtype=bool)
        self._build()
        from scipy import ndimage
        self.edt = ndimage.distance_transform_edt(~self.grid) * self.res

    def _rect(self, x0, y0, x1, y1):
        i0, j0 = int(x0 / self.res), int(y0 / self.res)
        i1, j1 = int(x1 / self.res), int(y1 / self.res)
        self.grid[max(0, j0):min(self.ny, j1), max(0, i0):min(self.nx, i1)] = True

    def _build(self):
        W, H, t = self.width, self.height, 0.3
        self._rect(0, 0, W, t); self._rect(0, H - t, W, H)
        self._rect(0, 0, t, H); self._rect(W - t, 0, W, H)
        self._rect(3.0, 3.0, 4.0, 5.2); self._rect(6.6, 1.0, 7.6, 3.2)
        self._rect(8.6, 4.6, 9.8, 6.6); self._rect(4.8, 5.6, 6.0, 6.8)

    def obstacle_rects(self):
        return [(3.0, 3.0, 4.0, 5.2), (6.6, 1.0, 7.6, 3.2),
                (8.6, 4.6, 9.8, 6.6), (4.8, 5.6, 6.0, 6.8)]

    def raycast(self, origin, abs_angles, max_range):
        step = self.res * 0.5
        n_steps = max(1, int(max_range / step))
        dirs = np.stack([np.cos(abs_angles), np.sin(abs_angles)], axis=1)
        s = np.arange(1, n_steps + 1) * step
        pts = np.asarray(origin)[None, None, :] + s[None, :, None] * dirs[:, None, :]
        ix = (pts[..., 0] / self.res).astype(np.intp)
        iy = (pts[..., 1] / self.res).astype(np.intp)
        inb = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        occ = np.zeros(ix.shape, dtype=bool)
        occ[inb] = self.grid[iy[inb], ix[inb]]
        stop = occ | ~inb
        has = stop.any(axis=1)
        first = np.where(has, stop.argmax(axis=1), n_steps)
        return np.where(has, (first + 1) * step, max_range).astype(np.float32)

    def clearance(self, p):
        ix, iy = int(p[0] / self.res), int(p[1] / self.res)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return float(self.edt[iy, ix])
        return 0.0


class AFSKinematics:
    """Front-referenced articulated kinematics."""
    def __init__(self, L_f=0.5, L_r=0.5, half_w=0.22, r_disc=0.28, g_max=0.75):
        self.L_f, self.L_r = L_f, L_r
        self.half_w, self.r_disc, self.g_max = half_w, r_disc, g_max

    def theta_f_dot(self, v_f, omega, g):
        return (self.L_r * omega + v_f * math.sin(g)) / (self.L_f * math.cos(g) + self.L_r)

    def integrate(self, state, v_f, omega, dt):
        xf, yf, th_f, g = state
        thd = self.theta_f_dot(v_f, omega, g)
        xf += dt * v_f * math.cos(th_f)
        yf += dt * v_f * math.sin(th_f)
        th_f = math.atan2(math.sin(th_f + dt * thd), math.cos(th_f + dt * thd))
        g = float(np.clip(g + dt * omega, -self.g_max, self.g_max))
        return np.array([xf, yf, th_f, g])

    def frames(self, state):
        xf, yf, th_f, g = state
        th_r = th_f - g
        F = np.array([xf, yf])
        ef = np.array([math.cos(th_f), math.sin(th_f)])
        er = np.array([math.cos(th_r), math.sin(th_r)])
        P = F - self.L_f * ef
        R = P - self.L_r * er
        return F, P, R, th_f, th_r

    def disc_centres(self, state):
        F, P, R, th_f, th_r = self.frames(state)
        ef = np.array([math.cos(th_f), math.sin(th_f)])
        er = np.array([math.cos(th_r), math.sin(th_r)])
        pts = [F - s * ef for s in (0.0, self.L_f / 2, self.L_f)]      # front body F..P
        pts += [P - t * er for t in (self.L_r / 2, self.L_r)]          # rear body  P..R
        return pts


class AFSSimNode(Node):
    def __init__(self):
        super().__init__("afs_sim")
        p = self.declare_parameter
        self.L_f = p("link_front", 0.5).value     # front axle -> hinge
        self.L_r = p("link_rear", 0.5).value      # hinge -> rear axle
        self.half_w = p("half_width", 0.22).value
        self.r_disc = p("disc_radius", 0.28).value
        self.g_max = p("gamma_max", 0.75).value
        self.v_max = p("v_max", 1.2).value
        self.omega_max = p("omega_max", 1.2).value
        self.sim_rate = p("sim_rate", 100.0).value
        self.scan_rate = p("scan_rate", 15.0).value
        self.n_rays = p("scan_rays", 540).value
        self.max_range = p("scan_range", 8.0).value
        self.arena_w = p("arena_width", 12.0).value
        self.arena_h = p("arena_height", 8.0).value
        start = p("start_pose", [1.3, 1.3, 0.0, 0.0]).value

        self.gt = GroundTruthMap(self.arena_w, self.arena_h, res=0.05)
        self.kin = AFSKinematics(self.L_f, self.L_r, self.half_w, self.r_disc, self.g_max)
        self.state = np.array(start, dtype=float)
        self.cmd = np.zeros(2)            # [v_f, omega]
        self.dt = 1.0 / self.sim_rate

        self.pub_scan = self.create_publisher(LaserScan, "scan", qos_profile_sensor_data)
        self.pub_odom = self.create_publisher(Odometry, "odom", 10)
        self.pub_art = self.create_publisher(Float64, "afs/articulation", 10)
        self.pub_col = self.create_publisher(Bool, "afs/collision", 10)
        self.pub_mark = self.create_publisher(MarkerArray, "afs/markers", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Float64MultiArray, "afs/cmd", self.cmd_cb, 10)
        self.create_subscription(Twist, "cmd_vel", self.twist_cb, 10)

        self.create_timer(self.dt, self.sim_step)
        self.create_timer(1.0 / self.scan_rate, self.scan_step)
        self.create_timer(1.0, self.publish_obstacle_markers)
        self.get_logger().info("AFS sim (front-referenced) up. cmd [v_f, omega] on /afs/cmd or /cmd_vel.")

    def cmd_cb(self, msg):
        if len(msg.data) >= 2:
            self.cmd = np.array([msg.data[0], msg.data[1]], float)

    def twist_cb(self, msg):
        self.cmd = np.array([msg.linear.x, msg.angular.z], float)

    def sim_step(self):
        v_f = float(np.clip(self.cmd[0], -self.v_max, self.v_max))
        omega = float(np.clip(self.cmd[1], -self.omega_max, self.omega_max))
        self.state = self.kin.integrate(self.state, v_f, omega, self.dt)
        now = self.get_clock().now().to_msg()
        F, P, R, th_f, th_r = self.kin.frames(self.state)
        g = float(self.state[3])

        od = Odometry()
        od.header.stamp = now; od.header.frame_id = "odom"; od.child_frame_id = "base_link"
        od.pose.pose.position.x = float(F[0]); od.pose.pose.position.y = float(F[1])
        od.pose.pose.orientation = yaw_to_quat(th_f)
        od.twist.twist.linear.x = v_f
        od.twist.twist.angular.z = self.kin.theta_f_dot(v_f, omega, g)
        self.pub_odom.publish(od)
        self.pub_art.publish(Float64(data=g))

        hit = any(self.gt.clearance(c) < self.r_disc for c in self.kin.disc_centres(self.state))
        self.pub_col.publish(Bool(data=bool(hit)))

        # TF: front axle is base_link; hinge is L_f behind it; rear_link rotated by -gamma
        self._tf(now, "odom", "base_link", F[0], F[1], th_f)
        self._tf(now, "base_link", "front_lidar", 0.0, 0.0, 0.0)
        self._tf(now, "base_link", "hinge", -self.L_f, 0.0, 0.0)
        self._tf(now, "hinge", "rear_link", 0.0, 0.0, -g)
        self.publish_body_markers(now, hit)

    def scan_step(self):
        F, P, R, th_f, th_r = self.kin.frames(self.state)
        rel = np.linspace(-math.pi, math.pi, self.n_rays, endpoint=False)
        ranges = self.gt.raycast(F, th_f + rel, self.max_range)
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = "front_lidar"
        msg.angle_min = float(rel[0]); msg.angle_max = float(rel[-1])
        msg.angle_increment = float(rel[1] - rel[0])
        msg.range_min = 0.0; msg.range_max = float(self.max_range)
        msg.ranges = ranges.tolist()
        self.pub_scan.publish(msg)

    def _tf(self, stamp, parent, child, x, y, yaw):
        t = TransformStamped()
        t.header.stamp = stamp; t.header.frame_id = parent; t.child_frame_id = child
        t.transform.translation.x = float(x); t.transform.translation.y = float(y)
        t.transform.rotation = yaw_to_quat(yaw)
        self.tf.sendTransform(t)

    def publish_body_markers(self, stamp, hit):
        ma = MarkerArray()
        # front body spans base_link origin (front axle) back to hinge (-L_f)
        # rear body spans hinge back to rear axle (-L_r) in rear_link
        for idx, (frame, length) in enumerate([("base_link", self.L_f), ("rear_link", self.L_r)]):
            m = Marker(); m.header.stamp = stamp; m.header.frame_id = frame
            m.ns = "body"; m.id = idx; m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position.x = -length / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = length; m.scale.y = 2 * self.half_w; m.scale.z = 0.3
            m.color.a = 0.9
            if hit:
                m.color.r = 0.9
            elif idx == 0:
                m.color.g, m.color.b = 0.8, 0.9     # front = cyan
            else:
                m.color.r, m.color.g = 1.0, 0.5     # rear = orange
            ma.markers.append(m)
        self.pub_mark.publish(ma)

    def publish_obstacle_markers(self):
        ma = MarkerArray()
        for i, (x0, y0, x1, y1) in enumerate(self.gt.obstacle_rects()):
            m = Marker(); m.header.stamp = self.get_clock().now().to_msg(); m.header.frame_id = "odom"
            m.ns = "obstacles"; m.id = i; m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position.x = (x0 + x1) / 2; m.pose.position.y = (y0 + y1) / 2
            m.pose.orientation.w = 1.0
            m.scale.x = (x1 - x0); m.scale.y = (y1 - y0); m.scale.z = 0.5
            m.color.a = 0.6; m.color.r = m.color.g = m.color.b = 0.5
            ma.markers.append(m)
        self.pub_mark.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = AFSSimNode()
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