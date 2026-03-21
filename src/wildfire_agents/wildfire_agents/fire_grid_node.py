#!/usr/bin/env python3
"""
ROS 2 node that simulates fire spread on a grid.

Replaces a Gazebo-Classic world plugin with a pure-ROS node so it works with
Gazebo Harmonic (gz sim) or entirely headless.

Publishes:  /fire_grid   (wildfire_msgs/msg/FireGrid)
Subscribes: /water_spray  (geometry_msgs/msg/Point)
"""

import math
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from wildfire_msgs.msg import FireGrid


class FireGridNode(Node):

    def __init__(self):
        super().__init__('fire_grid_node')

        self.declare_parameter('grid_size', 50)
        self.declare_parameter('cell_size', 1.0)
        self.declare_parameter('base_spread_prob', 0.05)
        self.declare_parameter('wind_factor', 2.0)
        self.declare_parameter('wind_direction', 45.0)
        self.declare_parameter('update_rate', 1.0)
        self.declare_parameter('initial_fire_radius', 2)
        self.declare_parameter('fire_start_x', 0.0)
        self.declare_parameter('fire_start_y', 12.0)

        self.grid_size = self.get_parameter('grid_size').value
        self.cell_size = self.get_parameter('cell_size').value
        self.base_spread_prob = self.get_parameter('base_spread_prob').value
        self.wind_factor = self.get_parameter('wind_factor').value
        self.wind_direction = self.get_parameter('wind_direction').value
        self.update_rate = self.get_parameter('update_rate').value
        self.initial_fire_radius = self.get_parameter('initial_fire_radius').value
        fire_start_x = self.get_parameter('fire_start_x').value
        fire_start_y = self.get_parameter('fire_start_y').value

        n = self.grid_size
        self.fire_intensity = [0.0] * (n * n)

        cx = int(fire_start_x / self.cell_size + n / 2.0)
        cy = int(fire_start_y / self.cell_size + n / 2.0)
        r = self.initial_fire_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < n and 0 <= y < n:
                        self.fire_intensity[y * n + x] = 1.0

        self._tick_count = 0

        self.fire_grid_pub = self.create_publisher(FireGrid, '/fire_grid', 10)
        self.create_subscription(Point, '/water_spray', self._water_spray_cb, 10)

        period = 1.0 / max(self.update_rate, 0.01)
        self.create_timer(period, self._tick)

        initial_burning = sum(1 for v in self.fire_intensity if v > 0.1)
        self.get_logger().info(
            f'[FIRE] Started: {n}x{n} grid, wind {self.wind_direction}°, '
            f'update {self.update_rate} Hz, blob radius {self.initial_fire_radius}, '
            f'initial_burning={initial_burning} cells')

    # -- callbacks -----------------------------------------------------------

    def _tick(self):
        self._tick_count += 1
        self._spread_fire()
        self._publish()

        if self._tick_count % 10 == 0:
            n = self.grid_size
            burning = sum(1 for v in self.fire_intensity if v > 0.1)
            max_int = max(self.fire_intensity) if self.fire_intensity else 0.0
            self.get_logger().info(
                f'[FIRE] tick={self._tick_count} '
                f'burning={burning}/{n*n} '
                f'max_intensity={max_int:.2f}')

    def _water_spray_cb(self, msg: Point):
        n = self.grid_size
        gx = int((msg.x + n * self.cell_size / 2.0) / self.cell_size)
        gy = int((msg.y + n * self.cell_size / 2.0) / self.cell_size)
        self.get_logger().info(
            f'[FIRE] Water spray at world=({msg.x:.1f}, {msg.y:.1f}) '
            f'grid=({gx}, {gy})')
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                self._extinguish(gx + dx, gy + dy, 0.3)

    # -- fire spread ---------------------------------------------------------

    def _spread_fire(self):
        n = self.grid_size
        new = list(self.fire_intensity)

        # W, E, S, N  (and their canonical angles)
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        angles = [180.0, 0.0, 270.0, 90.0]

        wind_rad = math.radians(self.wind_direction)

        for y in range(n):
            for x in range(n):
                idx = y * n + x
                if self.fire_intensity[idx] < 0.1:
                    continue
                for i in range(4):
                    nx_, ny_ = x + dx[i], y + dy[i]
                    if not (0 <= nx_ < n and 0 <= ny_ < n):
                        continue
                    nidx = ny_ * n + nx_
                    if self.fire_intensity[nidx] >= 0.1:
                        continue

                    prob = self.base_spread_prob
                    angle_diff = abs(angles[i] - self.wind_direction)
                    if angle_diff > 180.0:
                        angle_diff = 360.0 - angle_diff
                    if angle_diff < 45.0:
                        prob *= self.wind_factor      # downwind boost
                    elif angle_diff > 135.0:
                        prob *= 0.5                    # upwind penalty

                    if random.random() < prob:
                        new[nidx] = min(1.0, self.fire_intensity[idx] * 0.8)

        self.fire_intensity = new

    # -- helpers -------------------------------------------------------------

    def _extinguish(self, x: int, y: int, amount: float):
        n = self.grid_size
        if 0 <= x < n and 0 <= y < n:
            idx = y * n + x
            self.fire_intensity[idx] = max(0.0, self.fire_intensity[idx] - amount)

    def _publish(self):
        msg = FireGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.width = self.grid_size
        msg.height = self.grid_size
        msg.cell_size = self.cell_size
        msg.intensity = self.fire_intensity
        self.fire_grid_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FireGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
