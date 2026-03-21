#!/usr/bin/env python3
"""
Lightweight simulated odometry for all robots.

Subscribes to /{name}/cmd_vel, integrates velocities via dead-reckoning,
and publishes /{name}/odometry at 20 Hz.  Replaces a full Gazebo physics
pipeline so the nav controllers get pose feedback without needing robot
models spawned in the simulator.
"""

import math
import os

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def _yaw_to_quaternion(yaw: float):
    """Return (x, y, z, w) quaternion for a rotation about Z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class _RobotState:
    __slots__ = ('x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz', 'wz')

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.wz = 0.0


class SimOdomNode(Node):

    def __init__(self):
        super().__init__('sim_odom_node')

        self.dt = 0.05  # 20 Hz
        self.states: dict[str, _RobotState] = {}
        self.odom_pubs = {}

        num_ff = int(os.environ.get('NUM_FIREFIGHTERS', '4'))

        robot_configs = [('scout', 0.0, 0.0, 0.0)]
        for i in range(num_ff):
            robot_configs.append((f'firefighter_{i + 1}', 0.0, -12.0, 0.0))

        for name, x0, y0, z0 in robot_configs:
            state = _RobotState(x=x0, y=y0, z=z0)
            self.states[name] = state

            self.odom_pubs[name] = self.create_publisher(
                Odometry, f'/{name}/odometry', 10)

            self.create_subscription(
                Twist, f'/{name}/cmd_vel',
                lambda msg, n=name: self._cmd_vel_cb(n, msg), 10)

        self.create_timer(self.dt, self._tick)

        names = [c[0] for c in robot_configs]
        self.get_logger().info(
            f'[SIM_ODOM] Started for {len(names)} robots: {names} '
            f'@ {1.0/self.dt:.0f} Hz')

    def _cmd_vel_cb(self, name: str, msg: Twist):
        s = self.states[name]
        s.vx = msg.linear.x
        s.vy = msg.linear.y
        s.vz = msg.linear.z
        s.wz = msg.angular.z

    def _tick(self):
        stamp = self.get_clock().now().to_msg()

        for name, s in self.states.items():
            s.yaw += s.wz * self.dt
            # Keep yaw in [-pi, pi]
            while s.yaw > math.pi:
                s.yaw -= 2 * math.pi
            while s.yaw < -math.pi:
                s.yaw += 2 * math.pi

            # Ground robots: body-frame forward (vx) rotated by yaw
            s.x += s.vx * math.cos(s.yaw) * self.dt
            s.y += s.vx * math.sin(s.yaw) * self.dt
            # Scout drone also uses linear.y (strafe) and linear.z (altitude)
            s.x += s.vy * math.cos(s.yaw + math.pi / 2) * self.dt
            s.y += s.vy * math.sin(s.yaw + math.pi / 2) * self.dt
            s.z += s.vz * self.dt

            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = 'world'
            odom.child_frame_id = f'{name}/base_link'

            odom.pose.pose.position.x = s.x
            odom.pose.pose.position.y = s.y
            odom.pose.pose.position.z = s.z

            qx, qy, qz, qw = _yaw_to_quaternion(s.yaw)
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw

            odom.twist.twist.linear.x = s.vx
            odom.twist.twist.linear.y = s.vy
            odom.twist.twist.linear.z = s.vz
            odom.twist.twist.angular.z = s.wz

            self.odom_pubs[name].publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = SimOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
