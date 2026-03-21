#!/usr/bin/env python3
"""
3D scene publisher for Foxglove Studio visualization.

Subscribes to /fire_grid and robot odometry, publishes 3D scene updates
via the foxglove-sdk (SceneUpdate) so Foxglove's 3D panel can render
procedural terrain, trees, rocks, fire, and robots — no foxglove_bridge
ROS package required.

Connects to the foxglove server started by foxglove_viz on port 8766.
"""

import math
import os
import time
import random as _random

import numpy as np

import foxglove
from foxglove import schemas as fgs

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from wildfire_msgs.msg import FireGrid


# ---------------------------------------------------------------------------
# Simple value-noise (no external deps) for terrain height
# ---------------------------------------------------------------------------

_PERM = list(range(256))
_random.Random(42).shuffle(_PERM)
_PERM *= 2


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a, b, t):
    return a + t * (b - a)


def _value_noise_2d(x, y):
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
    val = 0.0
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        val += amp * _value_noise_2d(x * freq, y * freq)
        amp *= gain
        freq *= lacunarity
    return val


# ---------------------------------------------------------------------------
# Foxglove helpers
# ---------------------------------------------------------------------------

def _fg_pose(x, y, z):
    return fgs.Pose(
        position=fgs.Vector3(x=x, y=y, z=z),
        orientation=fgs.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _fg_color(r, g, b, a=1.0):
    return fgs.Color(r=r, g=g, b=b, a=a)


def _fg_vec3(x, y, z):
    return fgs.Vector3(x=x, y=y, z=z)


def _fg_ts():
    t = time.time_ns()
    return fgs.Timestamp(sec=t // 1_000_000_000, nsec=t % 1_000_000_000)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERRAIN_NOISE_SCALE = 0.25
TERRAIN_HEIGHT_AMP = 0.6
TREE_COUNT = 100
TREE_SEED = 12345
ROCK_COUNT = 30
ROCK_SEED = 67890
FRAME_ID = 'world'


class ScenePublisher3D(Node):

    def __init__(self):
        super().__init__('scene_publisher_3d')

        num_ff = int(os.environ.get('NUM_FIREFIGHTERS', '4'))
        self._grid_size = 50
        self._cell_size = 1.0
        self._half_world = self._grid_size * self._cell_size / 2.0

        foxglove.start_server(port=8766)

        # --- terrain height cache ---
        self._terrain_h = self._build_terrain_heights()

        # --- procedural geometry (built once) ---
        self._tree_positions = []
        self._tree_data = []
        self._terrain_entity = self._build_terrain_entity()
        self._tree_entity = self._build_tree_entity()
        self._rock_entity = self._build_rock_entity()

        # publish static scene once, then re-publish slowly for late joiners
        self._publish_static()
        self.create_timer(5.0, self._publish_static)

        # --- robot state ---
        self._ff_pos = {}
        self._ff_water = {}
        self._scout_pos = (0.0, -12.0, 5.0)

        for i in range(num_ff):
            fid = f'firefighter_{i + 1}'
            self._ff_pos[fid] = (0.0, -12.0, 0.0)
            self._ff_water[fid] = 100.0
            self.create_subscription(
                Odometry, f'/{fid}/odometry',
                lambda msg, f=fid: self._odom_cb(f, msg), 10)
            self.create_subscription(
                Float32, f'/{fid}/water_level',
                lambda msg, f=fid: self._water_cb(f, msg), 10)

        self.create_subscription(
            Odometry, '/scout/odometry', self._scout_odom_cb, 10)

        self._last_fire = None
        self.create_subscription(FireGrid, '/fire_grid', self._fire_cb, 10)

        self.create_timer(0.2, self._publish_robots)

        self.get_logger().info(
            f'ScenePublisher3D started — ws://localhost:8766 — {num_ff} firefighters, '
            f'{TREE_COUNT} trees, {ROCK_COUNT} rocks')

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
                h[gy, gx] = _fbm(wx * TERRAIN_NOISE_SCALE, wy * TERRAIN_NOISE_SCALE) * TERRAIN_HEIGHT_AMP
        return h

    def _build_terrain_entity(self):
        n = self._grid_size
        cs = self._cell_size
        cubes = []
        for gy in range(n):
            for gx in range(n):
                h = float(self._terrain_h[gy, gx])
                wx = (gx - n / 2.0 + 0.5) * cs
                wy = (gy - n / 2.0 + 0.5) * cs
                tile_h = max(0.05, abs(h) + 0.05)

                t = float(np.clip((h / TERRAIN_HEIGHT_AMP + 1.0) / 2.0, 0.0, 1.0))
                gr = 0.25 + 0.35 * (1.0 - t)
                gg = 0.40 + 0.30 * (1.0 - t)
                gb = 0.15 + 0.10 * t

                cubes.append(fgs.CubePrimitive(
                    pose=_fg_pose(wx, wy, h / 2.0),
                    size=_fg_vec3(cs * 0.98, cs * 0.98, tile_h),
                    color=_fg_color(gr, gg, gb),
                ))

        return fgs.SceneEntity(
            timestamp=_fg_ts(), frame_id=FRAME_ID, id='terrain',
            lifetime=fgs.Duration(sec=0, nsec=0),
            frame_locked=True, cubes=cubes,
        )

    # -----------------------------------------------------------------------
    # Trees
    # -----------------------------------------------------------------------

    def _build_tree_entity(self):
        rng = _random.Random(TREE_SEED)
        n = self._grid_size
        cs = self._cell_size
        half = self._half_world
        min_dist = 1.2

        positions = []
        attempts = 0
        while len(positions) < TREE_COUNT and attempts < TREE_COUNT * 20:
            attempts += 1
            wx = rng.uniform(-half + 0.5, half - 0.5)
            wy = rng.uniform(-half + 0.5, half - 0.5)
            if math.hypot(wx, wy) < 2.0:
                continue
            too_close = any(math.hypot(wx - px, wy - py) < min_dist for px, py, _ in positions)
            if too_close:
                continue
            gx = max(0, min(int((wx + half) / cs), n - 1))
            gy = max(0, min(int((wy + half) / cs), n - 1))
            gz = float(self._terrain_h[gy, gx])
            positions.append((wx, wy, gz))

        self._tree_positions = positions

        cylinders = []
        spheres = []
        tree_data = []
        for wx, wy, gz in positions:
            trunk_h = rng.uniform(1.0, 2.5)
            canopy_r = rng.uniform(0.5, 1.0)
            tree_data.append((trunk_h, canopy_r))

            cylinders.append(fgs.CylinderPrimitive(
                pose=_fg_pose(wx, wy, gz + trunk_h / 2.0),
                size=_fg_vec3(0.15, 0.15, trunk_h),
                color=_fg_color(0.36, 0.25, 0.13),
            ))

            shade = rng.uniform(0.0, 1.0)
            spheres.append(fgs.SpherePrimitive(
                pose=_fg_pose(wx, wy, gz + trunk_h + canopy_r * 0.6),
                size=_fg_vec3(canopy_r * 2.0, canopy_r * 2.0, canopy_r * 1.6),
                color=_fg_color(
                    0.10 + 0.15 * shade,
                    0.35 + 0.30 * (1.0 - shade),
                    0.05 + 0.10 * shade,
                ),
            ))

        self._tree_data = tree_data
        return fgs.SceneEntity(
            timestamp=_fg_ts(), frame_id=FRAME_ID, id='trees',
            lifetime=fgs.Duration(sec=0, nsec=0),
            frame_locked=True, cylinders=cylinders, spheres=spheres,
        )

    # -----------------------------------------------------------------------
    # Rocks
    # -----------------------------------------------------------------------

    def _build_rock_entity(self):
        rng = _random.Random(ROCK_SEED)
        n = self._grid_size
        cs = self._cell_size
        half = self._half_world
        spheres = []

        for _ in range(ROCK_COUNT):
            wx = rng.uniform(-half + 0.5, half - 0.5)
            wy = rng.uniform(-half + 0.5, half - 0.5)
            if math.hypot(wx, wy) < 1.5:
                continue
            gx = max(0, min(int((wx + half) / cs), n - 1))
            gy = max(0, min(int((wy + half) / cs), n - 1))
            gz = float(self._terrain_h[gy, gx])

            rx = rng.uniform(0.2, 0.6)
            ry = rng.uniform(0.2, 0.6)
            rz = rng.uniform(0.15, 0.35)
            grey = rng.uniform(0.35, 0.55)
            spheres.append(fgs.SpherePrimitive(
                pose=_fg_pose(wx, wy, gz + rz / 2.0),
                size=_fg_vec3(rx, ry, rz),
                color=_fg_color(grey, grey * 0.95, grey * 0.90),
            ))

        return fgs.SceneEntity(
            timestamp=_fg_ts(), frame_id=FRAME_ID, id='rocks',
            lifetime=fgs.Duration(sec=0, nsec=0),
            frame_locked=True, spheres=spheres,
        )

    # -----------------------------------------------------------------------
    # Publish static scene
    # -----------------------------------------------------------------------

    def _publish_static(self):
        foxglove.log('/tf', fgs.FrameTransform(
            timestamp=_fg_ts(),
            parent_frame_id='',
            child_frame_id='world',
            translation=fgs.Vector3(x=0.0, y=0.0, z=0.0),
            rotation=fgs.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ))
        foxglove.log('/scene', fgs.SceneUpdate(entities=[
            self._terrain_entity,
            self._tree_entity,
            self._rock_entity,
        ]))

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
    # Fire (updated every fire grid tick)
    # -----------------------------------------------------------------------

    def _publish_fire(self, msg: FireGrid):
        n = msg.width
        cs = msg.cell_size
        half = n * cs / 2.0
        ts = _fg_ts()

        fire_cubes = []
        burn_spheres = []

        for gy in range(n):
            for gx in range(n):
                val = msg.intensity[gy * n + gx]
                if val < 0.01:
                    continue

                wx = (gx - n / 2.0 + 0.5) * cs
                wy = (gy - n / 2.0 + 0.5) * cs
                gz = float(self._terrain_h[gy, gx]) if gy < self._grid_size and gx < self._grid_size else 0.0
                fire_h = max(0.05, val * 2.0)

                fire_cubes.append(fgs.CubePrimitive(
                    pose=_fg_pose(wx, wy, gz + fire_h / 2.0),
                    size=_fg_vec3(cs * 0.9, cs * 0.9, fire_h),
                    color=_fg_color(1.0, max(0.0, 0.6 * (1.0 - val)), 0.0, 0.65 + 0.30 * val),
                ))

        # burning tree canopies
        for i, (wx, wy, gz) in enumerate(self._tree_positions):
            gx = int((wx + half) / cs)
            gy = int((wy + half) / cs)
            if not (0 <= gx < n and 0 <= gy < n):
                continue
            val = msg.intensity[gy * n + gx]
            if val < 0.05:
                continue
            trunk_h, canopy_r = self._tree_data[i]
            burn_spheres.append(fgs.SpherePrimitive(
                pose=_fg_pose(wx, wy, gz + trunk_h + canopy_r * 0.6),
                size=_fg_vec3(canopy_r * 2.2, canopy_r * 2.2, canopy_r * 1.8),
                color=_fg_color(0.9 * val + 0.1, 0.3 * (1.0 - val), 0.0, 0.8),
            ))

        fire_entity = fgs.SceneEntity(
            timestamp=ts, frame_id=FRAME_ID, id='fire',
            lifetime=fgs.Duration(sec=2, nsec=0),
            cubes=fire_cubes, spheres=burn_spheres,
        )
        foxglove.log('/scene', fgs.SceneUpdate(entities=[fire_entity]))

    # -----------------------------------------------------------------------
    # Robot markers (5 Hz)
    # -----------------------------------------------------------------------

    def _publish_robots(self):
        ts = _fg_ts()

        cubes = []
        cylinders = []
        texts = []

        # water supply
        cylinders.append(fgs.CylinderPrimitive(
            pose=_fg_pose(0.0, 0.0, 0.5),
            size=_fg_vec3(1.0, 1.0, 1.0),
            color=_fg_color(0.2, 0.5, 0.9, 0.85),
        ))
        texts.append(fgs.TextPrimitive(
            pose=_fg_pose(0.0, 0.0, 1.3),
            billboard=True, font_size=14.0, scale_invariant=True,
            color=_fg_color(0.5, 0.8, 1.0),
            text='WATER',
        ))

        # firefighters
        for fid, (fx, fy, fz) in self._ff_pos.items():
            water_pct = self._ff_water.get(fid, 100.0)
            color = _fg_color(1.0, 0.85, 0.0) if water_pct > 20 else _fg_color(1.0, 0.4, 0.1)
            cubes.append(fgs.CubePrimitive(
                pose=_fg_pose(fx, fy, fz + 0.25),
                size=_fg_vec3(0.6, 0.5, 0.3),
                color=color,
            ))
            texts.append(fgs.TextPrimitive(
                pose=_fg_pose(fx, fy, fz + 0.8),
                billboard=True, font_size=12.0, scale_invariant=True,
                color=_fg_color(1.0, 1.0, 1.0),
                text=f'{fid} ({water_pct:.0f}%)',
            ))

        # scout drone
        sx, sy, sz = self._scout_pos
        sz = max(sz, 5.0)
        cubes.append(fgs.CubePrimitive(
            pose=_fg_pose(sx, sy, sz),
            size=_fg_vec3(0.4, 0.4, 0.1),
            color=_fg_color(0.2, 0.5, 1.0),
        ))
        texts.append(fgs.TextPrimitive(
            pose=_fg_pose(sx, sy, sz + 0.5),
            billboard=True, font_size=12.0, scale_invariant=True,
            color=_fg_color(0.8, 0.9, 1.0),
            text='Scout Drone',
        ))

        robot_entity = fgs.SceneEntity(
            timestamp=ts, frame_id=FRAME_ID, id='robots',
            lifetime=fgs.Duration(sec=1, nsec=0),
            cubes=cubes, cylinders=cylinders, texts=texts,
        )
        foxglove.log('/scene', fgs.SceneUpdate(entities=[robot_entity]))


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
