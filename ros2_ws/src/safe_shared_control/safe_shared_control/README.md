# AFS test simulator (ROS2 Humble)

`afs_sim_node.py` is a **plant-only** simulator for an articulated (frame-steered)
vehicle. It integrates the kinematics against a **static, pre-built ground-truth
map** (standard ROS `map_server` .yaml + .pgm) and publishes state + that map.
It does **no** safety filtering, and it does **no** LIDAR simulation — your
separate controller/CBF node closes the loop using the map directly.

```
   your controller node ──(/afs/cmd)──▶  afs_sim_node  ──(/map, /odom, /afs/articulation)──▶ controller
```

> Previously this simulated a 2D LIDAR raycast against the ground truth and
> streamed `/scan`, which the controller turned into an occupancy grid online.
> That's gone: the map is fully known up front and is published **once**
> (latched via `TRANSIENT_LOCAL` QoS) on `/map`, so the controller has the
> whole obstacle layout immediately instead of building it up as it drives.

## Static map format

Standard ROS `map_server` map: a `.yaml` file plus a `.pgm` image.

```yaml
image: default_map.pgm     # relative to this yaml, or absolute
resolution: 0.05           # metres / pixel
origin: [0.0, 0.0, 0.0]    # x, y, yaw of the map's bottom-left cell
negate: 0                  # 0: white=free/black=occupied, 1: inverted
occupied_thresh: 0.65
free_thresh: 0.196
```

A default map reproducing the old hardcoded arena (12m x 8m, 4 obstacle
rects) ships at `maps/default_map.yaml` / `.pgm`. Regenerate or customize it
with `generate_default_map.py` (edit the `RECTS` list, or point it at your
own layout), or just draw/export your own `.pgm` from any image editor /
SLAM run and write a matching `.yaml`.

`afs_sim_node.py` looks for the map, in order:
1. `map_yaml` parameter, if set.
2. `maps/default_map.yaml` next to the script (standalone run).
3. `maps/default_map.yaml` under the installed package's share directory
   (`ros2 run` — see Packaging below for the `setup.py` wiring needed).

## Topic interface

**Sim subscribes (you publish the command):**

| topic       | type                          | meaning                              |
|-------------|-------------------------------|--------------------------------------|
| `/afs/cmd`  | `std_msgs/Float64MultiArray`  | `data = [v, gamma_dot]` (preferred)  |
| `/cmd_vel`  | `geometry_msgs/Twist`         | `linear.x = v`, `angular.z = gamma_dot` (for teleop) |

**Sim publishes (your controller consumes):**

| topic                | type                          | notes                          |
|----------------------|-------------------------------|--------------------------------|
| `/map`               | `nav_msgs/OccupancyGrid`      | static ground truth, latched (TRANSIENT_LOCAL), published once at startup + re-sent every 2s |
| `/odom`              | `nav_msgs/Odometry`           | front axle pose, `odom→base_link` |
| `/afs/articulation`  | `std_msgs/Float64`            | articulation angle γ [rad]     |
| `/afs/collision`     | `std_msgs/Bool`               | true while a body overlaps GT  |
| `/afs/markers`       | `visualization_msgs/MarkerArray` | vehicle body only — obstacles are shown via RViz's **Map** display on `/map` |

**TF tree:** `odom → base_link → hinge → rear_link`

Your controller node's job: subscribe to `/map` (with a `TRANSIENT_LOCAL` QoS
so you get the latched copy even if you start after the sim), `/odom`, and
`/afs/articulation`; build CBF constraints straight from the map; and publish
`[v, gamma_dot]` on `/afs/cmd`. The vehicle geometry it needs (link lengths a,
b; body width; disc radius) matches the sim's parameters below.

Both `controller_node.py` and `afs_mpc_controller_node.py` already implement
this — they cache the received map once (no incremental mapping) and echo it
back out on `/afs/ogm` so existing RViz configs pointed at that topic keep
working.

## Run

Source ROS2 Humble first, then either:

```bash
# standalone (quickest) -- uses maps/default_map.yaml next to the script
python3 afs_sim_node.py

# sanity-drive it by hand (publishes /cmd_vel)
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Parameters (override with `--ros-args -p name:=value`)

`link_rear` (0.5), `link_front` (0.5), `half_width` (0.22), `disc_radius` (0.28),
`gamma_max` (0.75), `v_max` (1.2), `gamma_dot_max` (1.2), `sim_rate` (100),
`map_yaml` (auto-detected `maps/default_map.yaml` if unset),
`start_pose` ([1.3,1.3,0,0]).

## RViz

`rviz2`, set **Fixed Frame = odom**, then add displays:
- **TF** (see the articulated frames move)
- **Map** on `/map` (the static ground-truth occupancy grid)
- **MarkerArray** on `/afs/markers` (the two body boxes)

## Packaging (for `ros2 run`)

Drop the node into a package, e.g. `afs_sim`:

```
afs_sim/
├── afs_sim/__init__.py
├── afs_sim/afs_sim_node.py
├── maps/
│   ├── default_map.yaml
│   └── default_map.pgm
├── package.xml
└── setup.py
```

`setup.py` entry point, plus installing the `maps/` directory so
`get_package_share_directory` can find it at runtime:

```python
import os
from glob import glob

setup(
    ...
    data_files=[
        ...
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    entry_points={'console_scripts': [
        'afs_sim = afs_sim.afs_sim_node:main',
    ]},
)
```

`package.xml` dependencies: `rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`,
`sensor_msgs`, `visualization_msgs`, `tf2_ros`, and `python3-yaml` (for
`map_yaml` parsing). Then:

```bash
colcon build --packages-select afs_sim
source install/setup.bash
ros2 run afs_sim afs_sim
```

> Note: requires a ROS2 Humble environment (rclpy). Pure-Python deps: numpy,
> scipy, pyyaml. The kinematics and map loading were validated offline, but
> run it in your ROS2 workspace to confirm the topics/TF against your controller.
