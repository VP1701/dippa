#!/usr/bin/env python3
"""
keyboard_teleop_node.py -- WASD keyboard -> sensor_msgs/Joy, video game style.

A small pygame window reads the keyboard state every tick: an axis is
driven while its key is held and returns to zero when it is released, and
keys combine (W + A drives and steers).  A terminal cannot do this, since
it only receives characters with the OS key repeat, never key releases.

  w / s : drive forward / reverse (hold)
  a / d : steer left / right (hold)
  c     : toggle assist      r : reset
  q, Esc or closing the window : quit

The window must have focus; without focus nothing counts as held.  Axes
ramp toward their target at `ramp` axis units per second.  Publishes Joy on
`topic` at `rate` Hz with axes = [steer, drive] and buttons = [assist,
reset], like the previous terminal version.

Needs pygame:  sudo apt install python3-pygame
"""
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Joy  # noqa: E402

BUTTONS = {pygame.K_c: 0, pygame.K_r: 1}          # Joy button index
QUIT_KEYS = (pygame.K_q, pygame.K_ESCAPE)
HELP = ("W/S drive   A/D steer", "C assist   R reset   Q quit")


def key_axis(pressed, positive, negative):
    """+1, -1 or 0 from two opposite keys; holding both cancels."""
    return float(pressed[positive]) - float(pressed[negative])


def ramp_toward(value, goal, step):
    return value + max(-step, min(step, goal - value))


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("keyboard_teleop")
        p = self.declare_parameter
        topic = p("topic", "/joy").value
        self.rate = float(p("rate", 50.0).value)
        self.ramp = float(p("ramp", 3.0).value)
        self.pub = self.create_publisher(Joy, topic, 10)
        self.drive = 0.0
        self.steer = 0.0
        self.quit = False

        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((360, 130))
        pygame.display.set_caption("AFS WASD teleop")
        self.font = pygame.font.Font(None, 26)
        self.create_timer(1.0 / self.rate, self.tick)

    def tick(self):
        buttons = [0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key in QUIT_KEYS:
                    self.quit = True
                elif event.key in BUTTONS:
                    buttons[BUTTONS[event.key]] = 1

        focused = pygame.key.get_focused()
        pressed = pygame.key.get_pressed()
        drive = key_axis(pressed, pygame.K_w, pygame.K_s) if focused else 0.0
        steer = key_axis(pressed, pygame.K_a, pygame.K_d) if focused else 0.0
        step = self.ramp / self.rate
        self.drive = ramp_toward(self.drive, drive, step)
        self.steer = ramp_toward(self.steer, steer, step)

        self.publish(buttons)
        self.draw(focused)

    def publish(self, buttons=(0, 0)):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [float(self.steer), float(self.drive)]
        msg.buttons = list(buttons)
        self.pub.publish(msg)

    def draw(self, focused):
        self.screen.fill((30, 30, 30))
        lines = HELP + (f"drive {self.drive:+.2f}   steer {self.steer:+.2f}",)
        for i, text in enumerate(lines):
            image = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(image, (12, 12 + 28 * i))
        if not focused:
            image = self.font.render("click here to drive", True,
                                     (255, 120, 80))
            self.screen.blit(image, (12, 96))
        pygame.display.flip()

    def stop(self):
        """Zero command, so the controller does not hold the last one."""
        self.drive = self.steer = 0.0
        self.publish()


def main():
    rclpy.init()
    node = KeyboardTeleop()
    try:
        while rclpy.ok() and not node.quit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():                  # after Ctrl-C ROS is already down
            node.stop()
        pygame.quit()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()