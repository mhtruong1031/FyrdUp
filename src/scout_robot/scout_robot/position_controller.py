#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry


class ScoutPositionController(Node):
    def __init__(self):
        super().__init__('scout_position_controller')

        self.declare_parameter('survey_altitude', 15.0)
        self.declare_parameter('kp_xy', 0.5)
        self.declare_parameter('kp_z', 0.8)

        self.survey_altitude = self.get_parameter('survey_altitude').value
        self.kp_xy = self.get_parameter('kp_xy').value
        self.kp_z = self.get_parameter('kp_z').value

        self.current_position = Point()
        self.target_position = Point(x=0.0, y=0.0, z=self.survey_altitude)
        self.has_reached_altitude = False

        self.cmd_vel_pub = self.create_publisher(Twist, '/scout/cmd_vel', 10)

        self.create_subscription(
            Odometry, '/scout/odometry', self._odom_cb, 10)
        self.create_subscription(
            Point, '/scout/target_position', self._target_cb, 10)

        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'Scout position controller started — altitude {self.survey_altitude}m')

    def _odom_cb(self, msg):
        self.current_position = msg.pose.pose.position

    def _target_cb(self, msg):
        self.target_position = msg
        self.get_logger().info(
            f'New target: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f})')

    def _control_loop(self):
        error_x = self.target_position.x - self.current_position.x
        error_y = self.target_position.y - self.current_position.y
        error_z = self.target_position.z - self.current_position.z

        if not self.has_reached_altitude and abs(error_z) < 0.5:
            self.has_reached_altitude = True
            self.get_logger().info('Reached survey altitude!')

        cmd = Twist()
        cmd.linear.x = max(-2.0, min(2.0, self.kp_xy * error_x))
        cmd.linear.y = max(-2.0, min(2.0, self.kp_xy * error_y))
        cmd.linear.z = max(-1.0, min(1.0, self.kp_z * error_z))

        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ScoutPositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
