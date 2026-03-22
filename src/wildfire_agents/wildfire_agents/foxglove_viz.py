#!/usr/bin/env python3
"""
Foxglove visualization node.

Subscribes to /fire_grid and firefighter topics, renders a bird's-eye image,
publishes it as sensor_msgs/Image on /viz/fire_image (consumed by ros_bridge
for VLM), and streams 2D viz to Foxglove Studio via the foxglove-sdk
WebSocket server (port **8765**). Scout reasoning Log channels are on **8766**
(see ``scene_publisher_3d``).
"""

import os

import cv2
import numpy as np
import foxglove
from foxglove import schemas as fgs

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from wildfire_msgs.msg import FireGrid
from cv_bridge import CvBridge

from .viz_colors import FIRE_DRAW_THRESHOLD, fire_cell_bgr

PX_PER_CELL = 16


class FoxgloveViz(Node):

    def __init__(self):
        super().__init__('foxglove_viz')

        num_ff = int(os.environ.get('NUM_FIREFIGHTERS', '4'))
        self.bridge = CvBridge()

        self.ff_positions = {}
        self.ff_water = {}

        for i in range(num_ff):
            ff_id = f'firefighter_{i + 1}'
            self.ff_positions[ff_id] = (0.0, -12.0)
            self.ff_water[ff_id] = 100.0

            self.create_subscription(
                Odometry, f'/{ff_id}/odometry',
                lambda msg, fid=ff_id: self._odom_cb(fid, msg), 10)
            self.create_subscription(
                Float32, f'/{ff_id}/water_level',
                lambda msg, fid=ff_id: self._water_cb(fid, msg), 10)

        self.create_subscription(FireGrid, '/fire_grid', self._fire_grid_cb, 10)

        self.image_pub = self.create_publisher(Image, '/viz/fire_image', 10)

        self.create_timer(1.0, self._publish_agent_status)

        foxglove.start_server(port=8765)
        self.get_logger().info(
            f'FoxgloveViz started — ws://localhost:8765 — {num_ff} firefighters')

    def _odom_cb(self, ff_id, msg):
        p = msg.pose.pose.position
        self.ff_positions[ff_id] = (p.x, p.y)

    def _water_cb(self, ff_id, msg):
        self.ff_water[ff_id] = msg.data

    def _fire_grid_cb(self, msg: FireGrid):
        n = msg.width
        px = PX_PER_CELL
        size = n * px

        img = np.full((size, size, 3), (34, 85, 34), dtype=np.uint8)

        burning_count = 0
        total_intensity = 0.0

        for y in range(n):
            for x in range(n):
                val = msg.intensity[y * n + x]
                if val > FIRE_DRAW_THRESHOLD:
                    burning_count += 1
                    total_intensity += val
                    y0, y1 = y * px + 1, (y + 1) * px - 1
                    x0, x1 = x * px + 1, (x + 1) * px - 1
                    img[y0:y1, x0:x1] = fire_cell_bgr(val)

        for i in range(n):
            img[i * px, :] = (60, 60, 60)
            img[:, i * px] = (60, 60, 60)

        cx = (n // 2) * px + px // 2
        cy = (n // 2) * px + px // 2
        cv2.rectangle(img, (cx - 8, cy - 8), (cx + 8, cy + 8), (50, 140, 255), -1)
        cv2.rectangle(img, (cx - 8, cy - 8), (cx + 8, cy + 8), (255, 255, 255), 1)

        for ff_id, (fx, fy) in self.ff_positions.items():
            gx = int(fx + n / 2.0)
            gy = int(fy + n / 2.0)
            if 0 <= gx < n and 0 <= gy < n:
                center = (gx * px + px // 2, gy * px + px // 2)
                cv2.circle(img, center, 8, (0, 255, 255), -1)
                cv2.circle(img, center, 8, (255, 255, 255), 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ros_img = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        ros_img.header.stamp = self.get_clock().now().to_msg()
        ros_img.header.frame_id = 'viz'
        self.image_pub.publish(ros_img)

        _, png_buf = cv2.imencode('.png', img)
        foxglove.log('/viz/fire_image', fgs.CompressedImage(
            data=png_buf.tobytes(),
            format='png',
            frame_id='viz',
        ))

        threat = 'low' if burning_count < 5 else ('medium' if burning_count < 15 else 'high')
        foxglove.log('/viz/fire_stats', {
            'burning_cells': burning_count,
            'total_intensity': round(total_intensity, 2),
            'threat_level': threat,
            'grid_size': n,
        })

    def _publish_agent_status(self):
        status = {}
        for ff_id in self.ff_positions:
            pos = self.ff_positions[ff_id]
            status[ff_id] = {
                'x': round(pos[0], 2),
                'y': round(pos[1], 2),
                'water': round(self.ff_water.get(ff_id, 0.0), 1),
            }
        foxglove.log('/viz/agent_status', status)


def main(args=None):
    rclpy.init(args=args)
    node = FoxgloveViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
