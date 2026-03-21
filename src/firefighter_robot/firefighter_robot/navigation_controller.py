#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry


class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')

        self.robot_name = self.get_namespace().strip('/')
        if not self.robot_name:
            self.robot_name = 'firefighter_1'

        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 1.0)
        self.declare_parameter('position_tolerance', 0.3)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.position_tolerance = self.get_parameter('position_tolerance').value

        self.current_position = Point()
        self.current_yaw = 0.0
        self.target_position = None

        self.cmd_vel_pub = self.create_publisher(
            Twist, f'/{self.robot_name}/cmd_vel', 10)

        self.create_subscription(
            Odometry, f'/{self.robot_name}/odometry',
            self._odom_cb, 10)
        self.create_subscription(
            Point, f'/{self.robot_name}/target_position',
            self._target_cb, 10)

        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'{self.robot_name} navigation controller initialized')

    def _odom_cb(self, msg):
        self.current_position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _target_cb(self, msg):
        self.target_position = msg
        self.get_logger().info(
            f'{self.robot_name} new target: ({msg.x:.2f}, {msg.y:.2f})')

    def _control_loop(self):
        if self.target_position is None:
            return

        dx = self.target_position.x - self.current_position.x
        dy = self.target_position.y - self.current_position.y
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)

        angle_error = target_angle - self.current_yaw
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        cmd = Twist()

        if distance < self.position_tolerance:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info(f'{self.robot_name} reached target')
            self.target_position = None
        elif abs(angle_error) > 0.2:
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed if angle_error > 0 else -self.angular_speed
        else:
            cmd.linear.x = min(self.linear_speed, distance)
            cmd.angular.z = 0.5 * angle_error

        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
