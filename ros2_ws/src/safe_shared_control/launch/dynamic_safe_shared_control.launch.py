"""
dynamic_safe_shared_control.launch.py
AFS shared-control bring-up on the neural network plant.

Same as safe_shared_control.launch.py, but the simulator is
dynamic_simulator_node (actuator network, 100 Hz) and the default
controller is the MPC that predicts with the MPC network
(dynamic_mpc_controller_node).  The kinematic controllers can be run
against the same plant for comparison.

Launch arguments
  controller:=dynamic_mpc|mpc|reactive   shared controller (default
                                          dynamic_mpc); mpc and reactive
                                          are the kinematic controllers
  mpc_model:=<path>             MPC network .pt; empty = the node's default
                                 (models/mpc_model_01.pt)
  sim_model:=<path>             plant network .pt; empty = the sim's default
                                 (models/sim_model.pt)
  input, assist, manual, study, participant, phase, map_yaml, rviz,
  rviz_config, v_max, omega_max, joy_dev:  as in safe_shared_control.launch.py

The dynamic MPC runs at 10 Hz with one SQP iteration per tick: a tick
takes about 40-50 ms, so 20 Hz would overrun.  mpc_dt * rate must be a
whole number; the node checks it at startup.

Examples
  ros2 launch safe_shared_control dynamic_safe_shared_control.launch.py
  ros2 launch safe_shared_control dynamic_safe_shared_control.launch.py \\
      input:=gamepad
  # kinematic MPC on the same plant, for comparison:
  ros2 launch safe_shared_control dynamic_safe_shared_control.launch.py \\
      controller:=mpc
  ros2 launch safe_shared_control dynamic_safe_shared_control.launch.py \\
      mpc_model:=/abs/path/to/mpc_model_02.pt

Confirm executable names with:  ros2 pkg executables safe_shared_control
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PKG = 'safe_shared_control'

EXE_SIM = 'dynamic_simulator_node'
EXE_CONTROLLER = {'dynamic_mpc': 'dynamic_mpc_controller_node',
                  'mpc': 'mpc_controller_node',
                  'reactive': 'controller_node'}
EXE_KEYBOARD = 'keyboard_teleop_node'
EXE_STUDY = 'study_node'

# axes 0/1 are LeftX / LeftY in both the SDL2 'joy' and 'joy_linux' nodes.
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

# Node parameters override the class defaults: tune the controllers here.
MPC_PARAMS = dict(horizon=12, mpc_dt=0.2, cbf_gamma=1.0, margin=0.1,
                  disc_radius=1.0, influence_radius=4.0, smooth_v=0.1,
                  control_rate=20.0)
# One SQP iteration per tick (real-time iteration) keeps the solve well
# inside the 100 ms period; 20 Hz would still overrun.
DYNAMIC_MPC_PARAMS = dict(MPC_PARAMS, control_rate=10.0, sqp_iters=1,
                          horizon=10)


def flag(context, name):
    value = LaunchConfiguration(name).perform(context).lower()
    return value in ('1', 'true', 'yes')


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    share = FindPackageShare(PKG).perform(context)
    mpc_model = arg('mpc_model') or os.path.join(share, 'models', 'tdnn_test2_12_chan.pt')
    sim_model = arg('sim_model') or mpc_model       

    input_dev = arg('input')
    controller = arg('controller').lower()
    manual = flag(context, 'manual')
    assist_default = flag(context, 'assist')
    study = flag(context, 'study')
    map_yaml = arg('map_yaml') or os.path.join(
        FindPackageShare(PKG).perform(context), 'maps',
        'test_maze_edited.yaml')

    if input_dev not in INPUT_PRESETS:
        raise RuntimeError(
            f"input must be one of {list(INPUT_PRESETS)}, got '{input_dev}'")
    preset = INPUT_PRESETS[input_dev]

    # start_pose is the FRONT axle [x, y, theta, gamma]
    sim_params = {'start_pose': [5.5, 24.0, 1.5708, 0.0],
                  'map_yaml': map_yaml,
                  'model_path': sim_model,
                  'v_max': float(arg('v_max')),
                  'omega_max': float(arg('omega_max')),
                  'gamma_max': 0.65}
    if arg('sim_model'):
        sim_params['model_path'] = arg('sim_model')
    nodes = [Node(package=PKG, executable=EXE_SIM, name='simulator_node',
                  output='screen', parameters=[sim_params])]

    if manual:
        # Total bypass: no joystick source and no controller, /cmd_vel only.
        nodes.append(Node(
            package='teleop_twist_keyboard',
            executable='teleop_twist_keyboard',
            name='manual_teleop', output='screen',
            prefix='xterm -title "MANUAL teleop (no CBF/MPC in the loop)" -e',
        ))
    else:
        if input_dev == 'gamepad':
            nodes.append(Node(
                package='joy', executable='joy_node', name='joy_node',
                output='screen',
                parameters=[{'device_id': int(arg('joy_dev')),
                             'deadzone': 0.05,
                             'autorepeat_rate': 20.0}],
            ))
        elif input_dev == 'ponsse':
            nodes.append(Node(package='ponsse_handles',
                              executable='handle_node',
                              name='handle_node', output='screen'))
        elif input_dev == 'keyboard':
            # pygame window: click it to give it keyboard focus
            nodes.append(Node(
                package=PKG, executable=EXE_KEYBOARD, name='keyboard_teleop',
                output='screen', parameters=[{'topic': '/joy'}]))

        ctrl_params = dict(preset)
        ctrl_params.update(v_max=float(arg('v_max')),
                           omega_max=float(arg('omega_max')),
                           joy_lpf_alpha=0.85,
                           assist_default=assist_default)
        if study:
            ctrl_params['assist_button'] = -1   # condition fixed for the run
        if controller == 'dynamic_mpc':
            ctrl_params.update(DYNAMIC_MPC_PARAMS)
            ctrl_params['mpc_model_path'] = mpc_model
        elif controller == 'mpc':
            ctrl_params.update(MPC_PARAMS)
        nodes.append(Node(
            package=PKG, executable=EXE_CONTROLLER[controller],
            name='shared_controller', output='screen',
            parameters=[ctrl_params],
        ))

    if study:
        nodes.append(Node(
            package=PKG, executable=EXE_STUDY, name='study', output='screen',
            parameters=[{
                'participant': arg('participant'),
                'phase': arg('phase'),
                'assist': assist_default and not manual,
                'condition': ('bypass' if manual else
                              'assist' if assist_default else 'manual'),
                'joy_topic': preset['joy_topic'],
                'link_front': 1.1059,
            }],
        ))

    if flag(context, 'rviz'):
        rviz_path = arg('rviz_config') or os.path.join(
            FindPackageShare(PKG).perform(context), 'rviz', 'afs_mpc.rviz')
        if os.path.exists(rviz_path):
            rviz_args = ['-d', rviz_path]
        else:
            print(f"[launch] WARNING: RViz config not found at {rviz_path}, "
                  "starting RViz with no config.")
            rviz_args = []
        nodes.append(Node(package='rviz2', executable='rviz2', name='rviz2',
                          arguments=rviz_args))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('controller', default_value='dynamic_mpc',
                              choices=list(EXE_CONTROLLER),
                              description='shared controller to run'),
        DeclareLaunchArgument('mpc_model', default_value='',
                              description='MPC network .pt; empty = node '
                                          'default'),
        DeclareLaunchArgument('sim_model', default_value='',
                              description='plant network .pt; empty = sim '
                                          'default'),
        DeclareLaunchArgument('input', default_value='keyboard',
                              choices=list(INPUT_PRESETS),
                              description='joystick source'),
        DeclareLaunchArgument('assist', default_value='true',
                              description='start with the filter on (true) '
                                          'or raw passthrough (false)'),
        DeclareLaunchArgument('manual', default_value='false',
                              description='total bypass: no controller, '
                                          'teleop_twist_keyboard on /cmd_vel'),
        DeclareLaunchArgument('study', default_value='false',
                              description='run the user-study node and lock '
                                          'the assist toggle'),
        DeclareLaunchArgument('participant', default_value='P00',
                              description='participant id in every result'),
        DeclareLaunchArgument('phase', default_value='trial',
                              choices=['practice', 'trial'],
                              description='logged phase label'),
        DeclareLaunchArgument('map_yaml', default_value='',
                              description='map_server .yaml; empty = the '
                                          'test maze'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value='',
                              description='.rviz config; empty = installed '
                                          'afs_mpc.rviz'),
        DeclareLaunchArgument('v_max', default_value='1.0',
                              description='max drive speed m/s; keep within '
                                          'the network training data'),
        DeclareLaunchArgument('omega_max', default_value='1.2',
                              description='max articulation rate rad/s'),
        DeclareLaunchArgument('joy_dev', default_value='0',
                              description='gamepad device id'),
        OpaqueFunction(function=launch_setup),
    ])