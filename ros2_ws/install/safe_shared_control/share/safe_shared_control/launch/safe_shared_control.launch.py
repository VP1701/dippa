"""
safe_shared_control.launch.py
AFS shared-control bring-up with switchable joystick input.

Launch arguments
  input:=ponsse|gamepad|keyboard|none   joystick source     (default ponsse)
  controller:=reactive|mpc      which shared controller     (default reactive)
  rviz:=true|false              start RViz                  (default true)
  v_max:=<float>                max drive speed m/s         (default 1.2)
  omega_max:=<float>            max articulation rate rad/s (default 1.2)
  joy_dev:=<int>                gamepad device id           (default 0)

Examples
  ros2 launch safe_shared_control safe_shared_control.launch.py
  ros2 launch safe_shared_control safe_shared_control.launch.py input:=keyboard v_max:=0.4 omega_max:=0.6
  ros2 launch safe_shared_control safe_shared_control.launch.py input:=gamepad
  ros2 launch safe_shared_control safe_shared_control.launch.py input:=gamepad controller:=mpc

Gamepad needs ROS's joy driver (SDL2-based, publishes /joy):
  sudo apt install ros-humble-joy
Keyboard (WASD) opens its own terminal window, which needs xterm:
  sudo apt install xterm

Confirm executable names with:  ros2 pkg executables safe_shared_control
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

PKG = 'safe_shared_control'

# --- executable names (from `ros2 pkg executables safe_shared_control`) ---
EXE_SIM      = 'simulator_node'
EXE_REACTIVE = 'controller_node'
EXE_MPC      = 'mpc_controller_node'      # <-- confirm/adjust to your console_scripts name
EXE_KEYBOARD = 'keyboard_teleop_node'     # <-- add this console_script + rebuild

# --- per-input presets passed to the controller node ---
# axes 0/1 are LeftX / LeftY in BOTH the SDL2 'joy' node and the evdev
# 'joy_linux' node, so the left-stick mapping below is driver-independent.
# Drive = left stick Y, steer = left stick X. Verify signs (see notes) and
# flip invert_* if a direction is reversed.
INPUT_PRESETS = {
    'ponsse': dict(joy_topic='/right_controller/joy',
                   axis_drive=0, axis_steer=1,
                   invert_drive=False, invert_steer=True, deadzone=0.06),
    'gamepad': dict(joy_topic='/joy',
                    axis_drive=1, axis_steer=0,
                    invert_drive=False, invert_steer=False, deadzone=0.10),
    'keyboard': dict(joy_topic='/joy',
                     axis_drive=1, axis_steer=0,
                     invert_drive=False, invert_steer=False, deadzone=0.05),
    'none': dict(joy_topic='/right_controller/joy',
                 axis_drive=0, axis_steer=1,
                 invert_drive=False, invert_steer=True, deadzone=0.06),
}


def launch_setup(context, *args, **kwargs):
    input_dev = LaunchConfiguration('input').perform(context)
    controller = LaunchConfiguration('controller').perform(context).lower()
    use_rviz = LaunchConfiguration('rviz').perform(context).lower() in ('1', 'true', 'yes')
    joy_dev = int(LaunchConfiguration('joy_dev').perform(context))
    v_max = float(LaunchConfiguration('v_max').perform(context))
    omega_max = float(LaunchConfiguration('omega_max').perform(context))

    if input_dev not in INPUT_PRESETS:
        raise RuntimeError(f"input must be one of {list(INPUT_PRESETS)}, got '{input_dev}'")
    preset = INPUT_PRESETS[input_dev]

    # start_pose is the FRONT axle [x, y, theta, gamma]; the body extends ~1.0 m
    # behind it, so keep x clear of the 0.3 m wall (rear axle = x - 1.0 at heading 0).
    nodes = [Node(package=PKG, executable=EXE_SIM, name='simulator_node', output='screen',
                  parameters=[{'start_pose': [2.0, 1.5, 0.0, 0.0]}])]

    # joystick source
    if input_dev == 'gamepad':
        nodes.append(Node(
            package='joy', executable='joy_node', name='joy_node', output='screen',
            parameters=[{
                'device_id': joy_dev,
                'deadzone': 0.05,
                # republish at 20 Hz so idle sticks still hold the command
                # (the controller has a joy watchdog that zeroes on silence)
                'autorepeat_rate': 20.0,
            }],
        ))
    elif input_dev == 'ponsse':
        nodes.append(Node(
            package='ponsse_controllers', executable='ponsse_controllers',
            name='ponsse_controllers', output='screen',
        ))
    elif input_dev == 'keyboard':
        # Spawn in its own terminal so it has keyboard focus / stdin (a node
        # launched normally by ros2 launch has no interactive TTY). Needs xterm:
        #   sudo apt install xterm
        # If xterm is unavailable, run instead in a plain terminal:
        #   ros2 run safe_shared_control keyboard_teleop_node
        nodes.append(Node(
            package=PKG, executable=EXE_KEYBOARD, name='keyboard_teleop',
            output='screen', prefix='xterm -title "AFS WASD teleop" -e',
            parameters=[{'topic': '/joy', 'decay_timeout': 0.3}]))
    # input:=none -> bring your own Joy publisher

    # shared controller (reads Joy, maps axes from the preset)
    is_mpc = controller == 'mpc'
    ctrl_params = dict(preset)
    ctrl_params.update(dict(v_max=v_max, omega_max=omega_max,   # speed / sensitivity
                            joy_lpf_alpha=0.85))                 # lighter input filter = crisper
    if is_mpc:
        # NOTE: these node params are what actually take effect (they override the
        # AFSMPC class defaults). Tune the controller HERE, not in the .py signature.
        ctrl_params.update(dict(horizon=12, mpc_dt=0.2, cbf_gamma=2.5,
                                margin=0.1, disc_radius=0.28, influence_radius=1.5,
                                smooth_v=0.1, control_rate=20.0,
                                ogm_width=18.0, ogm_height=12.0))
    nodes.append(Node(
        package=PKG, executable=(EXE_MPC if is_mpc else EXE_REACTIVE),
        name='shared_controller', output='screen', parameters=[ctrl_params],
    ))

    # visualization
    if use_rviz:
        rviz_path = LaunchConfiguration('rviz_config').perform(context)
        if not rviz_path:
            rviz_path = os.path.join(FindPackageShare(PKG).perform(context),
                                     'rviz', 'afs_mpc.rviz')
        if not os.path.exists(rviz_path):
            print(f"[launch] WARNING: RViz config not found at:\n  {rviz_path}\n"
                  f"  -> add  (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz'))\n"
                  f"     to setup.py data_files and rebuild, OR pass\n"
                  f"     rviz_config:=/abs/path/to/afs_mpc.rviz .  Starting RViz with no config.")
            rviz_args = []
        else:
            rviz_args = ['-d', rviz_path]
        nodes.append(Node(package='rviz2', executable='rviz2', name='rviz2',
                          arguments=rviz_args))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('input', default_value='ponsse',
                              choices=['ponsse', 'gamepad', 'keyboard', 'none'],
                              description='joystick source'),
        DeclareLaunchArgument('controller', default_value='reactive',
                              choices=['reactive', 'mpc'],
                              description='which shared controller to run'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value='',
                              description='absolute path to a .rviz config; '
                                          'empty = use the installed package config'),
        DeclareLaunchArgument('v_max', default_value='1.2',
                              description='max drive speed m/s (lower = slower/less sensitive)'),
        DeclareLaunchArgument('omega_max', default_value='1.2',
                              description='max articulation rate rad/s (lower = gentler steering)'),
        DeclareLaunchArgument('joy_dev', default_value='0',
                              description='gamepad device id (input:=gamepad)'),
        OpaqueFunction(function=launch_setup),
    ])
