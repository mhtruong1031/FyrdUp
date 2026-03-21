#!/usr/bin/env python3
"""
Bird's-eye visualization renderer.

Subscribes to /fire_grid and firefighter topics, renders a top-down image,
and publishes it as sensor_msgs/Image on /viz/fire_image so the scout VLM
(and any external viewer) can consume it.

This replaces the Foxglove-based visualization with a pure ROS 2 node.
"""

import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from wildfire_msgs.msg import FireGrid
from cv_bridge import CvBridge

PX_PER_CELL = 32


class VizRenderer(Node):

    def __init__(self):
        super().__init__('viz_renderer')

        num_ff = int(os.environ.get('NUM_FIREFIGHTERS', '4'))
        self.bridge = CvBridge()

        self.ff_positions = {}
        self.ff_water = {}

        for i in range(num_ff):
            ff_id = f'firefighter_{i + 1}'
            self.ff_positions[ff_id] = (0.0, 0.0)
            self.ff_water[ff_id] = 100.0

            self.create_subscription(
                Odometry, f'/{ff_id}/odometry',
                lambda msg, fid=ff_id: self._odom_cb(fid, msg), 10)
            self.create_subscription(
                Float32, f'/{ff_id}/water_level',
                lambda msg, fid=ff_id: self._water_cb(fid, msg), 10)

        self.create_subscription(FireGrid, '/fire_grid', self._fire_grid_cb, 10)

        self.image_pub = self.create_publisher(Image, '/viz/fire_image', 10)

        self.get_logger().info(
            f'VizRenderer started — {num_ff} firefighters tracked')

    # -- callbacks -----------------------------------------------------------

    def _odom_cb(self, ff_id, msg):
        p = msg.pose.pose.position
        self.ff_positions[ff_id] = (p.x, p.y)

    def _water_cb(self, ff_id, msg):
        self.ff_water[ff_id] = msg.data

    def _fire_grid_cb(self, msg: FireGrid):
        n = msg.width
        px = PX_PER_CELL
        size = n * px

        # Dark green background (unburned terrain)
        img = np.full((size, size, 3), (34, 85, 34), dtype=np.uint8)

        for y in range(n):
            for x in range(n):
                val = msg.intensity[y * n + x]
                if val > 0.01:
                    r = min(255, int(val * 255))
                    g = max(0, int((1.0 - val) * 80))
                    y0, y1 = y * px + 1, (y + 1) * px - 1
                    x0, x1 = x * px + 1, (x + 1) * px - 1
                    img[y0:y1, x0:x1] = (r, g, 0)

        # Grid lines
        for i in range(n + 1):
            coord = i * px
            if coord < size:
                img[coord, :] = (60, 60, 60)
                img[:, coord] = (60, 60, 60)

        # Water supply at grid center
        cx = (n // 2) * px + px // 2
        cy = (n // 2) * px + px // 2
        half = 10
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half),
                       (255, 180, 50), -1)
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half),
                       (255, 255, 255), 1)

        # Firefighter positions (yellow circles)
        for ff_id, (fx, fy) in self.ff_positions.items():
            gx = int(fx + n / 2.0)
            gy = int(fy + n / 2.0)
            if 0 <= gx < n and 0 <= gy < n:
                center = (gx * px + px // 2, gy * px + px // 2)
                water_pct = self.ff_water.get(ff_id, 100.0)
                color = (0, 255, 255) if water_pct > 20 else (0, 100, 255)
                cv2.circle(img, center, 8, color, -1)
                cv2.circle(img, center, 8, (255, 255, 255), 1)

        # Publish as ROS Image
        ros_img = self.bridge.cv2_to_imgmsg(img, encoding='rgb8')
        ros_img.header.stamp = self.get_clock().now().to_msg()
        ros_img.header.frame_id = 'viz'
        self.image_pub.publish(ros_img)


def main(args=None):
    rclpy.init(args=args)
    node = VizRenderer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
