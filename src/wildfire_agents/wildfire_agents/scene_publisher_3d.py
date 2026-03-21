#!/usr/bin/env python3
"""
3D scene publisher for Foxglove Studio visualization.

Subscribes to /fire_grid and robot odometry, publishes MarkerArrays that
Foxglove's 3D panel renders natively.  Terrain and trees are procedurally
generated once at startup; fire and robot markers update in real time.
"""

import math
import os
import random as _random

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA, Float32
from visualization_msgs.msg import Marker, MarkerArray
from wildfire_msgs.msg import FireGrid
from tf2_ros import StaticTransformBroadcaster


# ---------------------------------------------------------------------------
# Simple value-noise (no external deps) for terrain height
# ---------------------------------------------------------------------------

_PERM = list(range(256))
_random.Random(42).shuffle(_PERM)
_PERM *= 2  # double for wrap-around


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a, b, t):
    return a + t * (b - a)


def _value_noise_2d(x, y):
    """Simple 2-D value noise in [0, 1]."""
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    xf = x - math.floor(x)
    yf = y - math.floor(y)
    u = _fade(xf)
    v = _fade(yf)
    aa = _PERM[_PERM[xi] + yi]
    ab = _PERM[_PERM[xi] + yi + 1]
    ba = _PERM[_PERM[xi + 1] + yi]
    bb = _PERM[_PERM[xi + 1] + yi + 1]
    x1 = _lerp(aa / 255.0, ba / 255.0, u)
    x2 = _lerp(ab / 255.0, bb / 255.0, u)
    return _lerp(x1, x2, v)


def _fbm(x, y, octaves=4, lacunarity=2.0, gain=0.5):
    """Fractal Brownian Motion layered on value noise."""
    val = 0.0
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        val += amp * _value_noise_2d(x * freq, y * freq)
        amp *= gain
        freq *= lacunarity
    return val


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def _duration(sec=0, nanosec=0):
    d = Duration()
    d.sec = sec
    d.nanosec = nanosec
    return d


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

TERRAIN_NOISE_SCALE = 0.25
TERRAIN_HEIGHT_AMP = 0.6
TREE_COUNT = 100
TREE_SEED = 12345
ROCK_COUNT = 30
ROCK_SEED = 67890


class ScenePublisher3D(Node):

    def __init__(self):
        super().__init__('scene_publisher_3d')

        num_ff = int(os.environ.get('NUM_FIREFIGHTERS', '4'))
        self._grid_size = 20
        self._cell_size = 1.0
        self._half_world = self._grid_size * self._cell_size / 2.0

        # --- publishers (latched via transient_local for static markers) ---
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._pub_terrain = self.create_publisher(
            MarkerArray, '/viz/terrain_markers', latched)
        self._pub_trees = self.create_publisher(
            MarkerArray, '/viz/tree_markers', latched)
        self._pub_rocks = self.create_publisher(
            MarkerArray, '/viz/rock_markers', latched)
        self._pub_fire = self.create_publisher(
            MarkerArray, '/viz/fire_markers', 10)
        self._pub_robots = self.create_publisher(
            MarkerArray, '/viz/robot_markers', 10)

        # --- static TF: world frame identity ---
        self._tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_tf()

        # --- terrain height cache (grid_size x grid_size) ---
        self._terrain_h = self._build_terrain_heights()

        # --- procedural geometry (built once) ---
        self._terrain_msg = self._build_terrain_markers()
        self._tree_positions = []
        self._tree_msg = self._build_tree_markers()
        self._rock_msg = self._build_rock_markers()

        # publish static markers once now (and re-publish on a slow timer for
        # late-joining Foxglove connections)
        self._publish_static()
        self.create_timer(5.0, self._publish_static)

        # --- robot state ---
        self._ff_pos = {}
        self._ff_water = {}
        self._scout_pos = (0.0, 0.0, 5.0)

        for i in range(num_ff):
            fid = f'firefighter_{i + 1}'
            self._ff_pos[fid] = (0.0, 0.0, 0.0)
            self._ff_water[fid] = 100.0
            self.create_subscription(
                Odometry, f'/{fid}/odometry',
                lambda msg, f=fid: self._odom_cb(f, msg), 10)
            self.create_subscription(
                Float32, f'/{fid}/water_level',
                lambda msg, f=fid: self._water_cb(f, msg), 10)

        self.create_subscription(
            Odometry, '/scout/odometry', self._scout_odom_cb, 10)

        # --- fire grid subscription ---
        self._last_fire = None
        self.create_subscription(FireGrid, '/fire_grid', self._fire_cb, 10)

        # timer for robot markers (5 Hz)
        self.create_timer(0.2, self._publish_robots)

        self.get_logger().info(
            f'ScenePublisher3D started — {num_ff} firefighters, '
            f'{TREE_COUNT} trees, {ROCK_COUNT} rocks')

    # -----------------------------------------------------------------------
    # Static TF
    # -----------------------------------------------------------------------

    def _publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'map'
        t.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(t)

    # -----------------------------------------------------------------------
    # Terrain generation
    # -----------------------------------------------------------------------

    def _build_terrain_heights(self):
        n = self._grid_size
        cs = self._cell_size
        h = np.zeros((n, n), dtype=np.float64)
        for gy in range(n):
            for gx in range(n):
                wx = (gx - n / 2.0 + 0.5) * cs
                wy = (gy - n / 2.0 + 0.5) * cs
                h[gy, gx] = _fbm(
                    wx * TERRAIN_NOISE_SCALE,
                    wy * TERRAIN_NOISE_SCALE,
                ) * TERRAIN_HEIGHT_AMP
        return h

    def _build_terrain_markers(self):
        ma = MarkerArray()
        n = self._grid_size
        cs = self._cell_size
        stamp = self.get_clock().now().to_msg()

        for gy in range(n):
            for gx in range(n):
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = 'world'
                m.ns = 'terrain'
                m.id = gy * n + gx
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.lifetime = _duration()

                h = float(self._terrain_h[gy, gx])
                wx = (gx - n / 2.0 + 0.5) * cs
                wy = (gy - n / 2.0 + 0.5) * cs

                m.pose.position.x = wx
                m.pose.position.y = wy
                m.pose.position.z = h / 2.0
                m.pose.orientation.w = 1.0

                tile_h = max(0.05, abs(h) + 0.05)
                m.scale.x = cs * 0.98
                m.scale.y = cs * 0.98
                m.scale.z = tile_h

                t = np.clip((h / TERRAIN_HEIGHT_AMP + 1.0) / 2.0, 0.0, 1.0)
                gr = 0.25 + 0.35 * (1.0 - t)
                gg = 0.40 + 0.30 * (1.0 - t)
                gb = 0.15 + 0.10 * t
                m.color = _rgba(gr, gg, gb)

                ma.markers.append(m)

        return ma

    # -----------------------------------------------------------------------
    # Tree generation (Poisson-disc-ish via rejection sampling)
    # -----------------------------------------------------------------------

    def _build_tree_markers(self):
        ma = MarkerArray()
        rng = _random.Random(TREE_SEED)
        n = self._grid_size
        cs = self._cell_size
        half = self._half_world
        stamp = self.get_clock().now().to_msg()
        min_dist = 1.2

        positions = []
        attempts = 0
        while len(positions) < TREE_COUNT and attempts < TREE_COUNT * 20:
            attempts += 1
            wx = rng.uniform(-half + 0.5, half - 0.5)
            wy = rng.uniform(-half + 0.5, half - 0.5)
            # keep away from the water supply at origin
            if math.hypot(wx, wy) < 2.0:
                continue
            too_close = False
            for px, py, _ in positions:
                if math.hypot(wx - px, wy - py) < min_dist:
                    too_close = True
                    break
            if too_close:
                continue
            gx = int((wx + half) / cs)
            gy = int((wy + half) / cs)
            gx = max(0, min(gx, n - 1))
            gy = max(0, min(gy, n - 1))
            ground_z = float(self._terrain_h[gy, gx])
            positions.append((wx, wy, ground_z))

        self._tree_positions = positions

        for i, (wx, wy, gz) in enumerate(positions):
            trunk_h = rng.uniform(1.0, 2.5)
            canopy_r = rng.uniform(0.5, 1.0)

            # trunk (cylinder)
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'world'
            m.ns = 'tree_trunk'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.lifetime = _duration()
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = gz + trunk_h / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = trunk_h
            m.color = _rgba(0.36, 0.25, 0.13)
            ma.markers.append(m)

            # canopy (sphere)
            m2 = Marker()
            m2.header.stamp = stamp
            m2.header.frame_id = 'world'
            m2.ns = 'tree_canopy'
            m2.id = i
            m2.type = Marker.SPHERE
            m2.action = Marker.ADD
            m2.lifetime = _duration()
            m2.pose.position.x = wx
            m2.pose.position.y = wy
            m2.pose.position.z = gz + trunk_h + canopy_r * 0.6
            m2.pose.orientation.w = 1.0
            m2.scale.x = canopy_r * 2.0
            m2.scale.y = canopy_r * 2.0
            m2.scale.z = canopy_r * 1.6
            shade = rng.uniform(0.0, 1.0)
            m2.color = _rgba(
                0.10 + 0.15 * shade,
                0.35 + 0.30 * (1.0 - shade),
                0.05 + 0.10 * shade,
            )
            ma.markers.append(m2)

        return ma

    # -----------------------------------------------------------------------
    # Rock generation
    # -----------------------------------------------------------------------

    def _build_rock_markers(self):
        ma = MarkerArray()
        rng = _random.Random(ROCK_SEED)
        n = self._grid_size
        cs = self._cell_size
        half = self._half_world
        stamp = self.get_clock().now().to_msg()

        for i in range(ROCK_COUNT):
            wx = rng.uniform(-half + 0.5, half - 0.5)
            wy = rng.uniform(-half + 0.5, half - 0.5)
            if math.hypot(wx, wy) < 1.5:
                continue
            gx = max(0, min(int((wx + half) / cs), n - 1))
            gy = max(0, min(int((wy + half) / cs), n - 1))
            gz = float(self._terrain_h[gy, gx])

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'world'
            m.ns = 'rocks'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.lifetime = _duration()
            rx = rng.uniform(0.2, 0.6)
            ry = rng.uniform(0.2, 0.6)
            rz = rng.uniform(0.15, 0.35)
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = gz + rz / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = rx
            m.scale.y = ry
            m.scale.z = rz
            grey = rng.uniform(0.35, 0.55)
            m.color = _rgba(grey, grey * 0.95, grey * 0.90)
            ma.markers.append(m)

        return ma

    # -----------------------------------------------------------------------
    # Publish helpers
    # -----------------------------------------------------------------------

    def _publish_static(self):
        self._pub_terrain.publish(self._terrain_msg)
        self._pub_trees.publish(self._tree_msg)
        self._pub_rocks.publish(self._rock_msg)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _odom_cb(self, fid, msg):
        p = msg.pose.pose.position
        self._ff_pos[fid] = (p.x, p.y, p.z)

    def _water_cb(self, fid, msg):
        self._ff_water[fid] = msg.data

    def _scout_odom_cb(self, msg):
        p = msg.pose.pose.position
        self._scout_pos = (p.x, p.y, p.z)

    def _fire_cb(self, msg: FireGrid):
        self._last_fire = msg
        self._publish_fire(msg)

    # -----------------------------------------------------------------------
    # Fire markers (updated every fire grid tick)
    # -----------------------------------------------------------------------

    def _publish_fire(self, msg: FireGrid):
        ma = MarkerArray()
        n = msg.width
        cs = msg.cell_size
        half = n * cs / 2.0
        stamp = self.get_clock().now().to_msg()
        mid = 0

        for gy in range(n):
            for gx in range(n):
                val = msg.intensity[gy * n + gx]
                if val < 0.01:
                    continue

                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = 'world'
                m.ns = 'fire'
                m.id = mid
                mid += 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.lifetime = _duration(sec=2)

                wx = (gx - n / 2.0 + 0.5) * cs
                wy = (gy - n / 2.0 + 0.5) * cs
                gz = float(self._terrain_h[gy, gx]) if gy < self._grid_size and gx < self._grid_size else 0.0
                fire_h = val * 2.0

                m.pose.position.x = wx
                m.pose.position.y = wy
                m.pose.position.z = gz + fire_h / 2.0
                m.pose.orientation.w = 1.0

                m.scale.x = cs * 0.9
                m.scale.y = cs * 0.9
                m.scale.z = max(0.05, fire_h)

                r = 1.0
                g = max(0.0, 0.6 * (1.0 - val))
                m.color = _rgba(r, g, 0.0, 0.65 + 0.30 * val)

                ma.markers.append(m)

        # Also update tree canopy colors to show burning trees
        self._publish_burning_trees(msg, stamp, ma)

        self._pub_fire.publish(ma)

    def _publish_burning_trees(self, msg: FireGrid, stamp, ma: MarkerArray):
        """Tint tree canopies in burning cells orange/black."""
        n = msg.width
        cs = msg.cell_size
        half = n * cs / 2.0

        for i, (wx, wy, gz) in enumerate(self._tree_positions):
            gx = int((wx + half) / cs)
            gy = int((wy + half) / cs)
            if not (0 <= gx < n and 0 <= gy < n):
                continue
            val = msg.intensity[gy * n + gx]
            if val < 0.05:
                continue

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'world'
            m.ns = 'tree_canopy_burn'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.lifetime = _duration(sec=2)
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = gz + 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = 1.2
            m.scale.y = 1.2
            m.scale.z = 1.0
            r = 0.9 * val + 0.1
            g = 0.3 * (1.0 - val)
            m.color = _rgba(r, g, 0.0, 0.8)
            ma.markers.append(m)

    # -----------------------------------------------------------------------
    # Robot markers (5 Hz)
    # -----------------------------------------------------------------------

    def _publish_robots(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        mid = 0

        # water supply
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = 'world'
        m.ns = 'water_supply'
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.lifetime = _duration(sec=1)
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.5
        m.pose.orientation.w = 1.0
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        m.color = _rgba(0.2, 0.5, 0.9, 0.85)
        ma.markers.append(m)

        # firefighters
        for fid, (fx, fy, fz) in self._ff_pos.items():
            # body cube
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'world'
            m.ns = 'ff_body'
            m.id = mid
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.lifetime = _duration(sec=1)
            m.pose.position.x = fx
            m.pose.position.y = fy
            m.pose.position.z = fz + 0.25
            m.pose.orientation.w = 1.0
            m.scale.x = 0.6
            m.scale.y = 0.5
            m.scale.z = 0.3
            water_pct = self._ff_water.get(fid, 100.0)
            if water_pct > 20:
                m.color = _rgba(1.0, 0.85, 0.0)
            else:
                m.color = _rgba(1.0, 0.4, 0.1)
            ma.markers.append(m)

            # label
            mt = Marker()
            mt.header.stamp = stamp
            mt.header.frame_id = 'world'
            mt.ns = 'ff_label'
            mt.id = mid
            mt.type = Marker.TEXT_VIEW_FACING
            mt.action = Marker.ADD
            mt.lifetime = _duration(sec=1)
            mt.pose.position.x = fx
            mt.pose.position.y = fy
            mt.pose.position.z = fz + 0.8
            mt.pose.orientation.w = 1.0
            mt.scale.z = 0.3
            mt.color = _rgba(1.0, 1.0, 1.0)
            mt.text = f'{fid} ({water_pct:.0f}%)'
            ma.markers.append(mt)

            mid += 1

        # scout drone
        sx, sy, sz = self._scout_pos
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = 'world'
        m.ns = 'scout'
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.lifetime = _duration(sec=1)
        m.pose.position.x = sx
        m.pose.position.y = sy
        m.pose.position.z = max(sz, 5.0)
        m.pose.orientation.w = 1.0
        m.scale.x = 0.4
        m.scale.y = 0.4
        m.scale.z = 0.1
        m.color = _rgba(0.2, 0.5, 1.0)
        ma.markers.append(m)

        # scout label
        mt = Marker()
        mt.header.stamp = stamp
        mt.header.frame_id = 'world'
        mt.ns = 'scout_label'
        mt.id = 0
        mt.type = Marker.TEXT_VIEW_FACING
        mt.action = Marker.ADD
        mt.lifetime = _duration(sec=1)
        mt.pose.position.x = sx
        mt.pose.position.y = sy
        mt.pose.position.z = max(sz, 5.0) + 0.5
        mt.pose.orientation.w = 1.0
        mt.scale.z = 0.3
        mt.color = _rgba(0.8, 0.9, 1.0)
        mt.text = 'Scout Drone'
        ma.markers.append(mt)

        self._pub_robots.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = ScenePublisher3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
