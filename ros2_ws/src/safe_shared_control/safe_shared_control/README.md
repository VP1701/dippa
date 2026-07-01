# AFS test simulator (ROS2 Humble)

`afs_sim_node.py` is a **plant-only** simulator for an articulated (frame-steered)
vehicle. It integrates the kinematics, fakes a 2D LIDAR against a ground-truth
map, and publishes state + sensors. It does **no** safety filtering — your
separate controller/CBF node closes the loop.

```
   your controller node ──(/afs/cmd)──▶  afs_sim_node  ──(/scan, /odom, /afs/articulation)──▶ controller
```

## Topic interface

**Sim subscribes (you publish the command):**

| topic       | type                          | meaning                              |
|-------------|-------------------------------|--------------------------------------|
| `/afs/cmd`  | `std_msgs/Float64MultiArray`  | `data = [v, gamma_dot]` (preferred)  |
| `/cmd_vel`  | `geometry_msgs/Twist`         | `linear.x = v`, `angular.z = gamma_dot` (for teleop) |

**Sim publishes (your controller consumes):**

| topic                | type                          | notes                          |
|----------------------|-------------------------------|--------------------------------|
| `/scan`              | `sensor_msgs/LaserScan`       | frame `front_lidar`, 360°      |
| `/odom`              | `nav_msgs/Odometry`           | rear axle pose, `odom→base_link` |
| `/afs/articulation`  | `std_msgs/Float64`            | articulation angle γ [rad]     |
| `/afs/collision`     | `std_msgs/Bool`               | true while a body overlaps GT  |
| `/afs/markers`       | `visualization_msgs/MarkerArray` | bodies + obstacles for RViz |

**TF tree:** `odom → base_link → hinge → front_link → front_lidar`
(the `hinge→front_link` rotation is the articulation angle, so the front body
and LIDAR move correctly in RViz).

Your controller node's job: read `/scan` (+ `/odom`, `/afs/articulation`), build
its occupancy grid / CBF constraints, and publish `[v, gamma_dot]` on `/afs/cmd`.
The vehicle geometry it needs (link lengths a, b; body width; disc radius) matches
the sim's parameters below.

## Run

Source ROS2 Humble first, then either:

```bash
# standalone (quickest)
python3 afs_sim_node.py

# sanity-drive it by hand (publishes /cmd_vel)
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Parameters (override with `--ros-args -p name:=value`)

`link_rear` (0.5), `link_front` (0.5), `half_width` (0.22), `disc_radius` (0.28),
`gamma_max` (0.75), `v_max` (1.2), `gamma_dot_max` (1.2), `sim_rate` (50),
`scan_rate` (15), `scan_rays` (180), `scan_range` (5.0),
`arena_width` (12), `arena_height` (8), `start_pose` ([1.3,1.3,0,0]).

## RViz

`rviz2`, set **Fixed Frame = odom**, then add displays:
- **TF** (see the articulated frames move)
- **LaserScan** on `/scan`
- **MarkerArray** on `/afs/markers` (the two body boxes + obstacles)

## Packaging (for `ros2 run`)

Drop the node into a package, e.g. `afs_sim`:

```
afs_sim/
├── afs_sim/__init__.py
├── afs_sim/afs_sim_node.py
├── package.xml
└── setup.py
```

`setup.py` entry point:

```python
entry_points={'console_scripts': [
    'afs_sim = afs_sim.afs_sim_node:main',
]},
```

`package.xml` dependencies: `rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`,
`sensor_msgs`, `visualization_msgs`, `tf2_ros`. Then:

```bash
colcon build --packages-select afs_sim
source install/setup.bash
ros2 run afs_sim afs_sim
```

> Note: requires a ROS2 Humble environment (rclpy). Pure-Python deps: numpy, scipy.
> The kinematics and LIDAR math were validated offline, but run it in your ROS2
> workspace to confirm the topics/TF against your controller.