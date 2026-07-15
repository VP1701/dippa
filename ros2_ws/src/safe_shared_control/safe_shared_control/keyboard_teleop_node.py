#!/usr/bin/env python3
"""
keyboard_teleop_node.py -- WASD keyboard -> sensor_msgs/Joy for the AFS controller.

Publishes Joy on `topic` (default /joy) at a steady rate, axes = [steer, drive]
in [-1, 1], so it feeds the shared controller exactly like a gamepad would (the
CBF/MPC assist still applies -- this does NOT bypass it).

  w / s : drive forward / reverse
  a / d : steer left / right
  c     : toggle CBF/MPC assist on/off (same as a gamepad's assist_button --
          useful for A/B testing whether the safety filter is doing anything:
          drive the same line with assist on, then off, and compare)
  space : stop immediately
  q     : quit

Hold a key to keep moving; release and it coasts to a stop. Must run in a
terminal with keyboard focus (the launch file opens one via `xterm -e`), so it
does not interfere with RViz.

If you want a TOTAL bypass of the controller/CBF/MPC (not just toggling the
assist flag inside it), skip this node and the shared controller entirely and
drive the sim directly via /cmd_vel instead, e.g.:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
The sim listens to /cmd_vel independently of /afs/cmd, with no safety filter
in that path at all. See `manual:=true` in the launch file.

Notes:
- `decay_timeout` (0.6 s) is set above the OS key-repeat *initial delay* so that
  holding a key gives continuous motion instead of stuttering. If holding still
  stutters, raise it; if it coasts too long after you release, lower it.
"""
import sys
import termios
import tty
import select
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

HELP = """
========== AFS keyboard teleop (WASD) ==========
  w / s : drive forward / reverse
  a / d : steer left / right
  space : stop      q : quit
  c     : toggle CBF/MPC assist on/off
  (type in THIS window; hold to move)
================================================
"""


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.declare_parameter('topic', '/joy')
        self.declare_parameter('rate', 50.0)
        self.declare_parameter('decay_timeout', 0.2)   # s without a key -> ramp to 0
        self.declare_parameter('ramp', 3.0)            # axis units / s toward target
        self.declare_parameter('assist_button', 0)     # which Joy.buttons[] slot to pulse
        topic = self.get_parameter('topic').value
        self.rate = float(self.get_parameter('rate').value)
        self.decay = float(self.get_parameter('decay_timeout').value)
        self.ramp = float(self.get_parameter('ramp').value)
        self.assist_button = int(self.get_parameter('assist_button').value)

        self.pub = self.create_publisher(Joy, topic, 10)
        self.drive = 0.0   # current (ramped) output
        self.steer = 0.0
        self.t_drive = 0.0  # target from last key
        self.t_steer = 0.0
        self.last_key = 0.0
        self.toggle_pending = False   # 'c' pressed since last tick -> pulse the button once
        self.lock = threading.Lock()
        self.alive = True

        self.reader = threading.Thread(target=self._read_keys, daemon=True)
        self.reader.start()
        self.create_timer(1.0 / self.rate, self._tick)
        sys.stdout.write(HELP)
        sys.stdout.flush()

    def _read_keys(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.alive:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                c = sys.stdin.read(1).lower()
                with self.lock:
                    self.last_key = time.time()
                    if c == 'w':
                        self.t_drive = 1.0
                    elif c == 's':
                        self.t_drive = -1.0
                    elif c == 'a':
                        self.t_steer = 1.0
                    elif c == 'd':
                        self.t_steer = -1.0
                    elif c in (' ', 'x'):
                        self.t_drive = 0.0
                        self.t_steer = 0.0
                    elif c == 'c':
                        self.toggle_pending = True
                    elif c in ('q', '\x03'):   # q or Ctrl-C
                        self.alive = False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        rclpy.try_shutdown()

    def _tick(self):
        now = time.time()
        with self.lock:
            if now - self.last_key > self.decay:   # released -> coast to stop
                self.t_drive = 0.0
                self.t_steer = 0.0
            td, ts = self.t_drive, self.t_steer
            pulse = self.toggle_pending
            self.toggle_pending = False
        step = self.ramp / self.rate
        self.drive += max(-step, min(step, td - self.drive))
        self.steer += max(-step, min(step, ts - self.steer))
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [float(self.steer), float(self.drive)]   # [steer, drive]
        # buttons[] sized so assist_button is always a valid index; pulse it
        # high for exactly one publish so the controller sees a rising edge.
        nbtn = max(1, self.assist_button + 1)
        msg.buttons = [0] * nbtn
        if pulse:
            msg.buttons[self.assist_button] = 1
            self.get_logger().info("assist toggle key pressed")
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = KeyboardTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.alive = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
