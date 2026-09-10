#!/usr/bin/env python3
"""
study_node.py -- goal sequencing, timing and result logging for the AFS
shared-control user study.

Shows one goal at a time in RViz. A goal counts as reached when the chosen
reference point of the machine (hinge by default) stays inside the goal disc
for `dwell` seconds -- and, for goals flagged in `goal_require_stop`, only
while the machine is essentially stopped. Reaching the last goal stops the
clock. Every goal writes a row to `results_file` immediately, so a crash or a
killed launch never loses a completed run.

The node owns no vehicle state and never publishes a command, so the exact
same node runs in the manual and the assisted condition.

INTERFACE
  Subscribes: /odom (Odometry), /afs/collision (Bool), /joy (Joy),
              /study/start (Empty), /afs/reset (Empty -- shared with the sim,
              so the controller's reset button resets the run too)
  Publishes:  /study/markers (MarkerArray -- goal disc, goal label, timer)

RUN
  ros2 run safe_shared_control study_node --ros-args \
      -p participant:=P03 -p phase:=trial -p assist:=false

  ros2 topic pub --once /study/start std_msgs/Empty {}
"""
import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, ColorRGBA, Empty
from visualization_msgs.msg import Marker, MarkerArray

IDLE, RUNNING, DONE = 0, 1, 2

# [x, y, radius] per goal, in the order they are presented.
DEFAULT_GOALS = [24.32, 15.15, 2.0,      # mid corridor, left
                 35.53, 14.05, 2.0,      # mid corridor, right
                 44.25, 25.25, 2.0,      # top tunnel, right end
                 17.19, 24.46, 2.0]      # top tunnel, left end -- finish
DEFAULT_NAMES = ['mid-left', 'mid-right', 'top-right', 'finish']

FIELDS = ['run_id', 'stamp', 'participant', 'phase', 'condition', 'assist',
          'goal_index', 'goal_name', 'split_s', 'elapsed_s', 'collisions',
          'event']


def quat_to_yaw(qz, qw):
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)


class StudyNode(Node):
    def __init__(self):
        super().__init__('study_node')
        p = self.declare_parameter
        self.participant = p('participant', 'P00').value
        self.phase = p('phase', 'trial').value            # practice | trial
        self.assist = bool(p('assist', True).value)
        self.condition = (p('condition', '').value
                          or ('assist' if self.assist else 'manual'))
        self.L_f = p('link_front', 1.1059).value
        self.reference = p('reference', 'hinge').value    # hinge | front
        self.dwell = p('dwell', 0.5).value
        self.stop_speed = p('stop_speed', 0.05).value
        self.auto_start = bool(p('auto_start', False).value)
        self.joy_topic = p('joy_topic', '/joy').value
        self.start_button = p('start_button', 2).value    # -1 to disable
        self.rate = p('publish_rate', 10.0).value
        self.results_file = os.path.expanduser(
            p('results_file', '~/afs_study/results.csv').value)

        flat = list(p('goals', DEFAULT_GOALS).value)
        self.goals = [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]
        names = list(p('goal_names', DEFAULT_NAMES).value)
        self.names = [names[i] if i < len(names) else f'goal{i + 1}'
                      for i in range(len(self.goals))]
        stops = list(p('goal_require_stop', [0, 0, 0, 1]).value)
        self.require_stop = [bool(stops[i]) if i < len(stops) else False
                             for i in range(len(self.goals))]

        self.state = IDLE
        self.k = 0
        self.t0 = None
        self.t_goal = None
        self.t_inside = None
        self.total = 0.0
        self.collisions = 0
        self.run_id = ''
        self.pose = None
        self.speed = 0.0
        self._col_prev = False
        self._btn_prev = 0
        self._flash = None

        self.pub = self.create_publisher(MarkerArray, 'study/markers', 1)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.create_subscription(Bool, 'afs/collision', self.collision_cb, 10)
        self.create_subscription(Joy, self.joy_topic, self.joy_cb, 10)
        self.create_subscription(Empty, 'study/start', self.start_cb, 10)
        self.create_subscription(Empty, 'afs/reset', self.reset_cb, 10)
        self.create_timer(1.0 / self.rate, self.tick)

        self.get_logger().info(
            f"study ready: participant={self.participant} phase={self.phase} "
            f"condition={self.condition} assist={self.assist}, "
            f"{len(self.goals)} goals -> {self.results_file}")

    # -------------------------------------------------------------- inputs
    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                     quat_to_yaw(q.z, q.w))
        self.speed = msg.twist.twist.linear.x
        if self.state == IDLE and self.auto_start and \
                abs(self.speed) > self.stop_speed:
            self.start()

    def collision_cb(self, msg):
        if msg.data and not self._col_prev and self.state == RUNNING:
            self.collisions += 1
            self.get_logger().warn(f'collision #{self.collisions}')
        self._col_prev = bool(msg.data)

    def joy_cb(self, msg):
        if not 0 <= self.start_button < len(msg.buttons):
            return
        b = msg.buttons[self.start_button]
        if b and not self._btn_prev:
            self.start()
        self._btn_prev = b

    def start_cb(self, msg):
        self.start()

    def reset_cb(self, msg):
        self.reset()

    # --------------------------------------------------------------- state
    def now(self):
        return self.get_clock().now()

    def elapsed(self):
        if self.t0 is None:
            return 0.0
        return (self.now() - self.t0).nanoseconds * 1e-9

    def start(self):
        if self.state == RUNNING:
            return
        self.run_id = (f'{self.participant}_'
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.state = RUNNING
        self.k = 0
        self.collisions = 0
        self.t0 = self.t_goal = self.now()
        self.t_inside = None
        self.get_logger().info(f'run {self.run_id} started')

    def reset(self):
        if self.state == RUNNING:
            self.write_row('abort')
        self.state = IDLE
        self.k = 0
        self.t0 = self.t_goal = self.t_inside = None
        self.collisions = 0
        self._flash = None
        self.get_logger().info('run reset')

    def ref_point(self):
        """Hinge (centre link) unless asked for the front axle. The hinge is
        roughly symmetric between the two bodies, so it treats a goal
        approached in reverse the same as one approached nose-first."""
        x, y, th = self.pose
        if self.reference == 'front':
            return x, y
        return x - self.L_f * math.cos(th), y - self.L_f * math.sin(th)

    def check_goal(self):
        gx, gy, gr = self.goals[self.k]
        px, py = self.ref_point()
        inside = math.hypot(px - gx, py - gy) < gr
        if inside and self.require_stop[self.k] and \
                abs(self.speed) > self.stop_speed:
            inside = False
        if not inside:
            self.t_inside = None
            return
        if self.t_inside is None:
            self.t_inside = self.now()
        elif (self.now() - self.t_inside).nanoseconds * 1e-9 >= self.dwell:
            self.advance()

    def advance(self):
        last = self.k == len(self.goals) - 1
        self.write_row('finish' if last else 'goal')
        gx, gy, gr = self.goals[self.k]
        self._flash = (gx, gy, gr, self.now())
        self.t_goal = self.now()
        self.t_inside = None
        if last:
            self.total = self.elapsed()
            self.state = DONE
            self.get_logger().info(
                f'FINISHED in {self.total:.1f} s, '
                f'collisions={self.collisions}')
        else:
            self.k += 1
            self.get_logger().info(f'goal {self.k}/{len(self.goals)} reached')

    # ------------------------------------------------------------- results
    def write_row(self, event):
        split = 0.0
        if self.t_goal is not None:
            split = (self.now() - self.t_goal).nanoseconds * 1e-9
        row = {
            'run_id': self.run_id,
            'stamp': datetime.now().isoformat(timespec='seconds'),
            'participant': self.participant,
            'phase': self.phase,
            'condition': self.condition,
            'assist': int(self.assist),
            'goal_index': self.k + 1,
            'goal_name': self.names[self.k],
            'split_s': round(split, 2),
            'elapsed_s': round(self.elapsed(), 2),
            'collisions': self.collisions,
            'event': event,
        }
        path = self.results_file
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

    # --------------------------------------------------------------- ticks
    def tick(self):
        if self.state == RUNNING and self.pose is not None:
            self.check_goal()
        self.publish_markers()

    def _marker(self, ns, mid, mtype):
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = self.now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def _status_text(self):
        if self.state == IDLE:
            return 'PRESS START', (0.6, 0.6, 0.6)
        if self.state == RUNNING:
            return (f'{self.elapsed():6.1f} s   GOAL {self.k + 1}/'
                    f'{len(self.goals)}: {self.names[self.k]}',
                    (1.0, 1.0, 1.0))
        return (f'FINISHED  {self.total:.1f} s   '
                f'collisions={self.collisions}', (0.1, 0.9, 0.1))

    def publish_markers(self):
        arr = MarkerArray()

        goal = self._marker('goal', 0, Marker.CYLINDER)
        label = self._marker('goal_label', 1, Marker.TEXT_VIEW_FACING)
        if self.state == RUNNING:
            gx, gy, gr = self.goals[self.k]
            goal.pose.position.x = float(gx)
            goal.pose.position.y = float(gy)
            goal.pose.position.z = 0.03
            goal.scale.x = goal.scale.y = 2.0 * float(gr)
            goal.scale.z = 0.06
            goal.color = ColorRGBA(r=1.0, g=0.25, b=0.25, a=0.45)
            label.pose.position.x = float(gx)
            label.pose.position.y = float(gy)
            label.pose.position.z = 1.2
            label.scale.z = 0.7
            label.color = ColorRGBA(r=1.0, g=0.35, b=0.35, a=1.0)
            label.text = f'{self.k + 1}'
        else:
            goal.action = Marker.DELETE
            label.action = Marker.DELETE
        arr.markers += [goal, label]

        flash = self._marker('goal_flash', 2, Marker.CYLINDER)
        if self._flash is not None and \
                (self.now() - self._flash[3]).nanoseconds * 1e-9 < 1.5:
            fx, fy, fr, _ = self._flash
            flash.pose.position.x = float(fx)
            flash.pose.position.y = float(fy)
            flash.pose.position.z = 0.04
            flash.scale.x = flash.scale.y = 2.0 * float(fr)
            flash.scale.z = 0.08
            flash.color = ColorRGBA(r=0.1, g=1.0, b=0.1, a=0.8)
        else:
            flash.action = Marker.DELETE
        arr.markers.append(flash)

        timer = self._marker('timer', 3, Marker.TEXT_VIEW_FACING)
        if self.pose is None:
            timer.action = Marker.DELETE
        else:
            txt, col = self._status_text()
            timer.pose.position.x = self.pose[0]
            timer.pose.position.y = self.pose[1]
            timer.pose.position.z = 2.6
            timer.scale.z = 0.5
            timer.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=1.0)
            timer.text = txt
            timer.lifetime = Duration(seconds=1.0).to_msg()
        arr.markers.append(timer)

        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = StudyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
