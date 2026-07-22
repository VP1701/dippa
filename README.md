# Thesis repo

## 3. Launch the sim + controller

Base command:
```bash
ros2 launch safe_shared_control safe_shared_control.launch.py \
  controller:=mpc input:=none
```

Common variations:

| Goal | Argument |
|---|---|
| Ponsse handles already running standalone | `input:=none` |
| Gamepad | `input:=gamepad` |
| Keyboard (WASD) | `input:=keyboard` (default) |
| Reactive CBF filter instead of MPC | `controller:=reactive` (default) |
| Custom map | `map_yaml:=/absolute/path/to/map.yaml` |
| Start with assist off (raw passthrough) | `assist:=false` |
| Total bypass, no controller node at all | `manual:=true` |
| Slower / gentler for testing | `v_max:=0.6 omega_max:=0.6` |
| No RViz | `rviz:=false` |

Full list of accepted arguments and their current defaults:
```bash
ros2 launch safe_shared_control safe_shared_control.launch.py --show-args
```

**Map arguments:**

- `map_yaml` takes an **absolute path** to a `map_server`-format `.yaml`
  (matching `.pgm` alongside it, per the `image:` field inside the yaml).
  Leave it unset/empty to use the sim's bundled default map — no argument
  needed for that case.
- Known map files on this machine:
  - Default arena (bundled, no `map_yaml` needed): open 12m×8m arena, 4
    obstacle rects.
  - Chicane / S-curve:
    ```
    map_yaml:=/home/veeti/dippa/ros2_ws/src/safe_shared_control/maps/chicane.yaml
    ```
- Full example with a custom map:
  ```bash
  ros2 launch safe_shared_control safe_shared_control.launch.py \
    controller:=mpc input:=none \
    map_yaml:=/home/veeti/dippa/ros2_ws/src/safe_shared_control/maps/chicane.yaml
  ```
- `start_pose` (where the vehicle spawns) is **not** exposed as a launch
  argument — it's a `simulator_node` parameter only, currently fixed per
  the launch file's own default for whichever map you pick. If you need a
  different spawn point on a custom map, that requires editing the launch
  file or overriding the node parameter directly rather than a `:=` arg.