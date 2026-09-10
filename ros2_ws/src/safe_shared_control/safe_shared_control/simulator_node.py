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

------------------------------------------------------------------- MAP
Ground truth is now a STATIC pre-built map (standard ROS map_server format:
a `.yaml` + `.pgm` pair) instead of a simulated LIDAR sweep. There is no
raycasting and no `/scan` topic any more -- the whole map is published once
(latched via TRANSIENT_LOCAL QoS) on `/map` as a nav_msgs/OccupancyGrid, and
the controller consumes that directly as its known-obstacle grid.

`map_yaml` (param) points at the map file. YAML fields follow map_server:
    image: <path to .pgm, relative to the yaml file unless absolute>
    resolution: <m/pixel>
    origin: [x, y, yaw]     # world pose of the pixel at row 0 (bottom of map)
    negate: 0 or 1
    occupied_thresh: 0..1
    free_thresh: 0..1
A bundled default map (`maps/default_map.yaml`) reproduces the previous
hardcoded arena so nothing else needs to change to try this out.

------------------------------------------------------------------ INTERFACE
Subscribes:  /afs/cmd  std_msgs/Float64MultiArray   data = [v_f, omega]
             /cmd_vel  geometry_msgs/Twist          linear.x = v_f, angular.z = omega
Publishes:   /map (nav_msgs/OccupancyGrid, latched, the static ground-truth map),
             /odom (nav_msgs/Odometry, odom->base_link at the FRONT axle),
             /afs/articulation (std_msgs/Float64, gamma),
             /afs/collision (std_msgs/Bool),
             /afs/markers (visualization_msgs/MarkerArray, vehicle body only --
                           obstacles are shown via RViz's Map display on /map)
TF: odom -> base_link(front axle) -> hinge -> rear_link
"""
import math
import os

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float64, Bool, Float64MultiArray, Empty
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from array import array

LATERAL = {
    'line': lambda s, amp, cyc: 0.0,
    'parabola': lambda s, amp, cyc: amp * 4.0 * s * (1.0 - s),
    'sine': lambda s, amp, cyc: amp * math.sin(2.0 * math.pi * cyc * s),
}
 
 
class MovingObstacle:
    """Circular obstacle travelling A -> B at constant speed.
 
    The path is the chord A->B plus a lateral offset:
 
        r(s) = a + s*L*e + lat(s)*n,    s in [0, 1]
 
    with e the unit tangent of the chord and n its left normal. s is a path
    parameter, not arc length, so a constant ds/dt would give a varying
    speed on any curved shape. Integrating
 
        ds/dt = direction * speed / ||r'(s)||
 
    instead gives ||dr/dt|| = speed exactly, for every shape. Since e and n
    are orthogonal, ||r'(s)|| = sqrt(L^2 + lat'(s)^2) >= L > 0, so the
    division is always well posed.
 
    Args:
        start, goal: (x, y) endpoints in map coordinates [m].
        speed: constant path speed [m/s].
        radius: obstacle radius [m].
        shape: key into LATERAL.
        amplitude: peak lateral offset [m]; ignored by 'line'.
        cycles: number of sine periods over one A->B traverse.
        mode: 'once' (stop at B), 'loop' (jump back to A), or 'pingpong'.
    """
 
    def __init__(self, start, goal, speed, radius=0.4, shape='line',
                 amplitude=0.0, cycles=1.0, mode='pingpong'):
        if shape not in LATERAL:
            raise ValueError(f"unknown shape {shape!r}, expected one of "
                             f"{sorted(LATERAL)}")
        if mode not in ('once', 'loop', 'pingpong'):
            raise ValueError(f"unknown mode {mode!r}, expected 'once', "
                             f"'loop' or 'pingpong'")
 
        self.a = np.asarray(start, dtype=float)
        self.b = np.asarray(goal, dtype=float)
        d = self.b - self.a
        self.L = float(np.linalg.norm(d))
        if self.L < 1e-9:
            raise ValueError("start and goal must differ")
 
        self.e = d / self.L
        self.n = np.array([-self.e[1], self.e[0]])
        self.speed = float(speed)
        self.radius = float(radius)
        self.amplitude = float(amplitude)
        self.cycles = float(cycles)
        self.shape = shape
        self.mode = mode
        self._lat = LATERAL[shape]
        self.reset()
 
    def reset(self):
        self.s = 0.0
        self.direction = 1.0
        self.done = False
 
    # ------------------------------------------------------------- geometry
    def _offset(self, s):
        return self._lat(s, self.amplitude, self.cycles)
 
    def _deriv(self, s, h=1e-4):
        """r'(s), central-differenced through the scalar offset function so a
        new shape needs no hand-derived derivative."""
        dlat = (self._offset(s + h) - self._offset(s - h)) / (2.0 * h)
        return self.L * self.e + dlat * self.n
 
    @property
    def position(self):
        return self.a + self.s * self.L * self.e + self._offset(self.s) * self.n
 
    @property
    def velocity(self):
        """Exact velocity vector; its magnitude is `speed` by construction."""
        if self.done:
            return np.zeros(2)
        d = self._deriv(self.s)
        return self.direction * self.speed * d / np.linalg.norm(d)
 
    # ---------------------------------------------------------- integration
    def _s_dot(self, s):
        return self.direction * self.speed / np.linalg.norm(self._deriv(s))
 
    def step(self, dt):
        """Advance the path parameter by one RK4 step of dt seconds."""
        if self.done:
            return
        k1 = self._s_dot(self.s)
        k2 = self._s_dot(self.s + 0.5 * dt * k1)
        k3 = self._s_dot(self.s + 0.5 * dt * k2)
        k4 = self._s_dot(self.s + dt * k3)
        self.s += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self._apply_mode()
 
    def _apply_mode(self):
        if 0.0 <= self.s <= 1.0:
            return
        if self.mode == 'once':
            self.s = float(np.clip(self.s, 0.0, 1.0))
            self.done = True
        elif self.mode == 'loop':
            self.s %= 1.0
        else:                                   # pingpong
            # Reflect rather than clamp: the path length covered this step is
            # preserved, so the speed stays constant through the turnaround.
            if self.s > 1.0:
                self.s = 2.0 - self.s
                self.direction = -1.0
            else:
                self.s = -self.s
                self.direction = 1.0
 
 
def disc_mask(radius, res):
    """Boolean stencil of a disc of `radius` m on a `res` m/cell grid."""
    rad = int(np.ceil(radius / res))
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    return (xx * xx + yy * yy) <= rad * rad
 
 
def stamp_disc(grid, mask, ic, jc, value=100):
    """Write `value` into `grid` where the disc stencil lands, clipped to the
    grid bounds. `ic, jc` are the (col, row) cell indices of the centre."""
    r = mask.shape[0] // 2
    ny, nx = grid.shape
    j0, j1 = max(0, jc - r), min(ny, jc + r + 1)
    i0, i1 = max(0, ic - r), min(nx, ic + r + 1)
    if j1 <= j0 or i1 <= i0:
        return
    sub = mask[j0 - jc + r:j1 - jc + r, i0 - ic + r:i1 - ic + r]
    grid[j0:j1, i0:i1][sub] = value


def yaw_to_quat(yaw):
    q = Quaternion(); q.z = math.sin(yaw * 0.5); q.w = math.cos(yaw * 0.5)
    return q


# --------------------------------------------------------------------------- #
# Static map loading (ROS map_server .yaml + .pgm)                            #
# --------------------------------------------------------------------------- #
def _read_pgm(path):
    """Minimal PGM reader (P5 binary or P2 ascii), no Pillow dependency."""
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        if magic not in (b'P5', b'P2'):
            raise ValueError(f"Unsupported PGM magic {magic!r} in {path} (need P5 or P2)")

        def _next_token():
            tok = b''
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError(f"Unexpected EOF reading PGM header of {path}")
                if c == b'#':
                    f.readline()
                    continue
                if c.isspace():
                    if tok:
                        return tok
                    continue
                tok += c

        width = int(_next_token())
        height = int(_next_token())
        maxval = int(_next_token())
        if magic == b'P5':
            dtype = np.uint8 if maxval < 256 else '>u2'
            nbytes = width * height * (1 if maxval < 256 else 2)
            data = np.frombuffer(f.read(nbytes), dtype=dtype)
        else:  # P2 ascii
            vals = f.read().split()
            data = np.array(vals[:width * height], dtype=np.int64)
        return data.reshape((height, width)).astype(np.float64), maxval


def load_ros_map(yaml_path):
    """Load a map_server-style map (yaml + pgm).

    Returns dict with:
      prob      (ny,nx) float64 in [0,1], occupancy probability. Row 0 is the
                BOTTOM of the map (y = origin_y), matching nav_msgs/OccupancyGrid.
      occupied  (ny,nx) bool  -- prob > occupied_thresh
      free      (ny,nx) bool  -- prob < free_thresh
      res       float, meters/pixel
      origin    (x, y) world coords of the bottom-left cell of the grid
      nx, ny    grid dimensions
    """
    yaml_path = os.path.abspath(yaml_path)
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
    base = os.path.dirname(yaml_path)
    image_path = meta['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(base, image_path)

    res = float(meta.get('resolution', 0.05))
    origin = meta.get('origin', [0.0, 0.0, 0.0])
    negate = int(meta.get('negate', 0))
    occ_th = float(meta.get('occupied_thresh', 0.65))
    free_th = float(meta.get('free_thresh', 0.196))

    raw, maxval = _read_pgm(image_path)
    # map_server: white pixels = free, black = occupied (unless negate=1)
    value = (raw / maxval) if negate else ((maxval - raw) / maxval)
    # file row 0 is the TOP of the image == max y == last row of the grid
    value = np.flipud(value)

    occupied = value > occ_th
    free = value < free_th
    ny, nx = value.shape
    return {'prob': value, 'occupied': occupied, 'free': free, 'res': res,
            'origin': (float(origin[0]), float(origin[1])), 'nx': nx, 'ny': ny}


def _default_map_path():
    """Look for the bundled sample map next to this script (standalone /
    `python3 afs_sim_node.py` use), then fall back to the installed package
    share directory (`ros2 run` use)."""
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, 'maps', 'default_map.yaml')
    if os.path.exists(local):
        return local
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('safe_shared_control')
        cand = os.path.join(share, 'maps', 'default_map.yaml')
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    return local  # will raise a clear FileNotFoundError downstream


# --------------------------------------------------------------------------- #
# Ground truth: static map + distance transform for collision checks          #
# --------------------------------------------------------------------------- #
class GroundTruthMap:
    def __init__(self, yaml_path, unknown_is_obstacle=True):
        m = load_ros_map(yaml_path)
        self.res = m['res']
        self.origin = np.array(m['origin'], float)
        self.ny, self.nx = m['ny'], m['nx']
        self.occupied = m['occupied']
        self.free = m['free']
        from scipy import ndimage
        # Blocked = occupied, plus (optionally) unknown. `free` is the explicit
        # free mask; anything not free is either occupied or unknown.
        blocked = self.occupied if not unknown_is_obstacle else ~self.free
        self.edt = ndimage.distance_transform_edt(~blocked) * self.res

    def world_to_cell(self, p):
        c = ((np.asarray(p) - self.origin) / self.res).astype(int)
        return int(c[0]), int(c[1])

    def clearance(self, p):
        ix, iy = self.world_to_cell(p)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return float(self.edt[iy, ix])
        return 0.0

    def occupancy_grid_data(self):
        data = np.full((self.ny, self.nx), -1, dtype=np.int8)
        data[self.free] = 0
        data[self.occupied] = 100
        return data


class AFSKinematics:
    """Front-referenced articulated kinematics."""
    def __init__(self, L_f=1.1059, L_r=0.985777778, half_w=0.9, r_disc=1.0, g_max=0.75):
        self.L_f, self.L_r = L_f, L_r
        self.half_w, self.r_disc, self.g_max = half_w, r_disc, g_max
        self.front_s = [x + L_f for x in (0.014, -0.553, -1.120)]
        self.rear_t = [0.307, -0.493, -1.293]

    def theta_f_dot(self, v_f, omega, g):
        return (self.L_r * omega + v_f * math.sin(g)) / (self.L_f * math.cos(g) + self.L_r)

    def integrate_euler(self, state, v_f, omega, dt):
        xf, yf, th_f, g = state
        at_limit = (g >= self.g_max and omega > 0) or (g <= -self.g_max and omega < 0)
        eff_omega = 0.0 if at_limit else omega
        thd = self.theta_f_dot(v_f, eff_omega, g)
        xf += dt * v_f * math.cos(th_f)
        yf += dt * v_f * math.sin(th_f)
        th_f = math.atan2(math.sin(th_f + dt * thd), math.cos(th_f + dt * thd))
        g = float(np.clip(g + dt * eff_omega, -self.g_max, self.g_max))
        return np.array([xf, yf, th_f, g])

    def f_c(self, state, v_f, omega):
        """Continuous kinematics qdot = f_c(q, u), no limits applied."""
        _, _, th_f, g = state
        return np.array([
            v_f * math.cos(th_f),
            v_f * math.sin(th_f),
            self.theta_f_dot(v_f, omega, g),
            omega,
        ])

    def integrate(self, state, v_f, omega, dt):
        """RK4 step. The articulation stop is applied by zeroing omega before
        integrating, not by clipping gamma afterwards: at the stop the actuator
        produces no motion, so it must not contribute its L_r*omega term to
        theta_f_dot either. Clipping after the fact would credit the heading
        with a joint rotation that never happened."""
        _, _, _, g = state
        at_limit = (g >= self.g_max and omega > 0) or (g <= -self.g_max and omega < 0)
        om = 0.0 if at_limit else omega

        k1 = self.f_c(state, v_f, om)
        k2 = self.f_c(state + 0.5 * dt * k1, v_f, om)
        k3 = self.f_c(state + 0.5 * dt * k2, v_f, om)
        k4 = self.f_c(state + dt * k3, v_f, om)
        out = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        out[2] = math.atan2(math.sin(out[2]), math.cos(out[2]))   # wrap heading
        out[3] = float(np.clip(out[3], -self.g_max, self.g_max))  # guard rounding
        return out

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
        """Six collision-disc centres, three per body, placed to cover the
        actual body rectangles rather than the axle-to-hinge links. Offsets
        are measured from the hinge along each body's own axis, matching the
        controller's disc list -- keep the two in sync or the collision check
        tests different geometry than the MPC constrains."""
        _, P, _, th_f, th_r = self.frames(state)
        ef = np.array([math.cos(th_f), math.sin(th_f)])
        er = np.array([math.cos(th_r), math.sin(th_r)])
        pts = [P + s * ef for s in self.front_s]
        pts += [P + t * er for t in self.rear_t]
        return pts


class AFSSimNode(Node):
    def __init__(self):
        super().__init__("afs_sim")
        p = self.declare_parameter
        self.L_f = p("link_front", 1.1059).value     # front axle -> hinge
        self.L_r = p("link_rear", 0.985777778).value      # hinge -> rear axle
        self.half_w = p("half_width", 0.9).value
        self.r_disc = p("disc_radius", 1.0).value
        self.g_max = p("gamma_max", 0.75).value
        self.v_max = p("v_max", 2.2).value
        self.omega_max = p("omega_max", 1.2).value
        self.sim_rate = p("sim_rate", 100.0).value
        self.map_yaml = p("map_yaml", _default_map_path()).value

        self.body_f_len = p("body_front_length", 1.7).value
        self.body_r_len = p("body_rear_length", 2.4).value
        self.body_f_cx = p("body_front_center_x", -self.L_f / 2).value
        self.body_r_cx = p("body_rear_center_x", -self.L_r / 2).value
        self.unknown_is_obstacle = p("unknown_is_obstacle", True).value

        start = p("start_pose", [6.0, 2.0, 1.5708, 0.0]).value

        # Moving obstacle pose
        self.obs_enable = p("obs_enable", True).value
        self.obs_start = p("obs_start", [51.0, 60.0]).value
        self.obs_goal = p("obs_goal", [60.0, 70.0]).value
        self.obs_speed = p("obs_speed", 0.5).value
        self.obs_radius = p("obs_radius", 0.4).value
        self.obs_shape = p("obs_shape", "line").value
        self.obs_amplitude = p("obs_amplitude", 0.0).value
        self.obs_cycles = p("obs_cycles", 1.0).value
        self.obs_mode = p("obs_mode", "pingpong").value
        self.map_rate = p("map_rate", 10.0).value


        # Save start pose
        self.state = np.array(start, dtype=float)
        self.start_pose = self.state.copy()

        self.gt = GroundTruthMap(self.map_yaml, self.unknown_is_obstacle)

        self.obs = None
        self._static_grid = self.gt.occupancy_grid_data()
        if self.obs_enable:
            self.obs = MovingObstacle(
                self.obs_start, self.obs_goal, self.obs_speed,
                self.obs_radius, self.obs_shape, self.obs_amplitude,
                self.obs_cycles, self.obs_mode)
            self._obs_mask = disc_mask(self.obs_radius, self.gt.res)


        self.kin = AFSKinematics(self.L_f, self.L_r, self.half_w, self.r_disc, self.g_max)
        self.state = np.array(start, dtype=float)
        self.cmd = np.zeros(2)            # [v_f, omega]
        self.dt = 1.0 / self.sim_rate

        map_qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map = self.create_publisher(OccupancyGrid, "map", map_qos)
        self.pub_odom = self.create_publisher(Odometry, "odom", 10)
        self.pub_art = self.create_publisher(Float64, "afs/articulation", 10)
        self.pub_col = self.create_publisher(Bool, "afs/collision", 10)
        self.pub_mark = self.create_publisher(MarkerArray, "afs/markers", 10)
        self.pub_obs = self.create_publisher(
            Float64MultiArray, "afs/moving_obstacle", 10)

        self.tf = TransformBroadcaster(self)
        self.create_subscription(Float64MultiArray, "afs/cmd", self.cmd_cb, 10)
        self.create_subscription(Twist, "cmd_vel", self.twist_cb, 10)
        self.create_subscription(Empty, "afs/reset", self.reset_cb, 10)

        self._map_msg = self._build_map_msg()
        self.pub_map.publish(self._map_msg)          # latched (transient local) copy

        map_period = 1.0 / self.map_rate if self.obs is not None else 2.0
        self.create_timer(map_period, self._republish_map)
 
        if self.obs is not None:
            mb = self.gt.nx * self.gt.ny / 1e6
            self.get_logger().info(
                f"Moving obstacle: {self.obs_shape} {self.obs_start} -> "
                f"{self.obs_goal}, {self.obs_speed} m/s, r={self.obs_radius} m, "
                f"mode={self.obs_mode}. Republishing /map at {self.map_rate} Hz "
                f"({mb:.2f} MB/msg, {mb * self.map_rate:.1f} MB/s).")


        self.create_timer(self.dt, self.sim_step)
        self.get_logger().info(
            f"AFS sim (front-referenced) up. Static map loaded from {self.map_yaml} "
            f"({self.gt.nx}x{self.gt.ny} @ {self.gt.res} m/cell). "
            "cmd [v_f, omega] on /afs/cmd or /cmd_vel.")

    def _obs_trajectory(self):
        """use this for integrator. need to finish"""
        d = obs_end_pose - obs_start_pos
        L = np.linalg.norm(d)
        tangent = d / L 

    def _build_map_msg(self):
        msg = OccupancyGrid()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(self.gt.res)
        msg.info.width = self.gt.nx
        msg.info.height = self.gt.ny
        msg.info.origin.position.x = float(self.gt.origin[0])
        msg.info.origin.position.y = float(self.gt.origin[1])
        msg.info.origin.orientation.w = 1.0
        # array('b') skips rclpy's per-element type check that .tolist() incurs
        # (~15x faster, and the difference matters once this runs at map_rate).
        msg.data = array('b', self._grid_with_obstacle().tobytes())
        return msg

    def _grid_with_obstacle(self):
        """Static occupancy grid with the moving obstacle stamped in as
        occupied. Returns the cached static grid untouched when disabled."""
        if self.obs is None:
            return self._static_grid
        grid = self._static_grid.copy()
        ic, jc = self.gt.world_to_cell(self.obs.position)
        stamp_disc(grid, self._obs_mask, ic, jc)
        return grid


    def reset_cb(self, msg):
        self.state = self.start_pose.copy()
        self.cmd = np.zeros(2)
        if self.obs is not None:
            self.obs.reset()
        self.get_logger().info(
            f"Reset -> state back to start pose {self.start_pose.tolist()}")

    def _republish_map(self):
        self._map_msg.header.stamp = self.get_clock().now().to_msg()
        if self.obs is not None:
            self._map_msg.data = array('b',
                                       self._grid_with_obstacle().tobytes())
        self.pub_map.publish(self._map_msg)

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


        # TF: front axle is base_link; hinge is L_f behind it; rear_link rotated by -gamma
        self._tf(now, "odom", "base_link", F[0], F[1], th_f)
        self._tf(now, "base_link", "hinge", -self.L_f, 0.0, 0.0)
        self._tf(now, "hinge", "rear_link", 0.0, 0.0, -g)
        self.publish_body_markers(now, hit)

    def _tf(self, stamp, parent, child, x, y, yaw):
        t = TransformStamped()
        t.header.stamp = stamp; t.header.frame_id = parent; t.child_frame_id = child
        t.transform.translation.x = float(x); t.transform.translation.y = float(y)
        t.transform.rotation = yaw_to_quat(yaw)
        self.tf.sendTransform(t)

    def publish_body_markers(self, stamp, hit):
        ma = MarkerArray()
        # Body boxes are drawn from explicit length/centre params, NOT from the
        # kinematic links: L_f/L_r are axle->hinge distances, which are shorter
        # than the actual bodies. body_*_cx is the box centre along -x in each
        # body frame (base_link: x=0 at front axle; rear_link: x=0 at hinge).
        bodies = [("base_link", self.body_f_len, self.body_f_cx),
                  ("rear_link", self.body_r_len, self.body_r_cx)]
        for idx, (frame, length, cx) in enumerate(bodies):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame
            m.ns = "body"
            m.id = idx
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(cx)
            m.pose.orientation.w = 1.0
            m.scale.x = float(length)
            m.scale.y = 2 * self.half_w
            m.scale.z = 0.3
            m.color.a = 0.9
            if hit:
                m.color.r = 0.9
            elif idx == 0:
                m.color.g, m.color.b = 0.8, 0.9     # front = cyan
            else:
                m.color.r, m.color.g = 1.0, 0.5     # rear = orange
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
