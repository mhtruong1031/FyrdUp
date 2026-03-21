#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry


class WaterManager(Node):
    def __init__(self):
        super().__init__('water_manager')

        self.robot_name = self.get_namespace().strip('/')
        if not self.robot_name:
            self.robot_name = 'firefighter_1'

        self.declare_parameter('tank_capacity', 100.0)
        self.declare_parameter('spray_rate', 5.0)
        self.declare_parameter('refill_rate', 20.0)
        self.declare_parameter('spray_radius', 1.5)
        self.declare_parameter('refill_radius', 2.0)

        self.tank_capacity = self.get_parameter('tank_capacity').value
        self.spray_rate = self.get_parameter('spray_rate').value
        self.refill_rate = self.get_parameter('refill_rate').value
        self.spray_radius = self.get_parameter('spray_radius').value
        self.refill_radius = self.get_parameter('refill_radius').value

        self.water_level = self.tank_capacity
        self.current_position = Point()
        self.is_spraying = False
        self.water_supply_position = Point(x=0.0, y=0.0, z=0.0)

        self.water_level_pub = self.create_publisher(
            Float32, f'/{self.robot_name}/water_level', 10)
        self.spray_point_pub = self.create_publisher(
            Point, '/water_spray', 10)

        self.create_subscription(
            Odometry, f'/{self.robot_name}/odometry',
            self._odom_cb, 10)
        self.create_subscription(
            Bool, f'/{self.robot_name}/spray_enable',
            self._spray_enable_cb, 10)

        self.create_timer(0.1, self._update_loop)
        self.create_timer(1.0, self._publish_status)

        self.get_logger().info(f'{self.robot_name} water manager initialized')

    def _odom_cb(self, msg):
        self.current_position = msg.pose.pose.position

    def _spray_enable_cb(self, msg):
        if msg.data and self.water_level > 0:
            self.is_spraying = True
        else:
            self.is_spraying = False

    def _update_loop(self):
        dt = 0.1

        dist_to_supply = math.sqrt(
            (self.current_position.x - self.water_supply_position.x) ** 2 +
            (self.current_position.y - self.water_supply_position.y) ** 2
        )

        if dist_to_supply < self.refill_radius and self.water_level < self.tank_capacity:
            self.water_level = min(
                self.tank_capacity,
                self.water_level + self.refill_rate * dt)
            if self.water_level >= self.tank_capacity:
                self.get_logger().info(f'{self.robot_name} tank full!')

        if self.is_spraying and self.water_level > 0:
            self.water_level = max(0.0, self.water_level - self.spray_rate * dt)

            spray_pt = Point(
                x=self.current_position.x,
                y=self.current_position.y,
                z=0.0)
            self.spray_point_pub.publish(spray_pt)

            if self.water_level == 0:
                self.is_spraying = False
                self.get_logger().warn(f'{self.robot_name} out of water!')

    def _publish_status(self):
        msg = Float32()
        msg.data = self.water_level
        self.water_level_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaterManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
