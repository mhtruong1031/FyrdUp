#!/usr/bin/env python3
"""
3D scene publisher for Foxglove Studio visualization.

Subscribes to /fire_grid and robot odometry, publishes 3D scene updates
via the foxglove-sdk (SceneUpdate) so Foxglove's 3D panel can render
terrain (from Depth Anything), trees, rocks, fire, and robots.

Starts its own Foxglove WebSocket server on port **8766** (same connection
as typed ``/scene`` / ``/tf``). Scout fast/slow reasoning is published here
as ``foxglove.Log`` on ``/scout_fast_reasoning`` and ``/scout_slow_reasoning``.
"""

import json
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
from std_msgs.msg import Float32, String
from wildfire_msgs.msg import FireGrid


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


def _fg_log_level_info():
    """``foxglove.Log.level`` must be ``LogLevel`` enum; SDK exposes ``Info`` (TitleCase), not ``INFO``/int."""
    return fgs.LogLevel.Info


def _fg_scout_log(message: str, name: str) -> fgs.Log:
    """Typed Log message — use with Foxglove **Log** panel like other MCAP channels."""
    return fgs.Log(
        timestamp=_fg_ts(),
        level=_fg_log_level_info(),
        message=message,
        name=name,
        file='',
        line=0,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERRAIN_HEIGHT_AMP = 2.0
TREE_COUNT = 100
TREE_SEED = 12345
ROCK_COUNT = 30
ROCK_SEED = 67890
FRAME_ID = 'world'
FOV_HALF_ANGLE_RAD = math.radians(35.0)


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

        self.create_subscription(
            String, '/scout/reasoning_status', self._scout_reasoning_ros_cb, 10)
        self.create_subscription(
            String, '/scout/decision_snapshot', self._scout_decision_ros_cb, 10)

        self.create_timer(0.2, self._publish_robots)

        self.get_logger().info(
            f'ScenePublisher3D started — ws://localhost:8766 — {num_ff} firefighters, '
            f'{TREE_COUNT} trees, {ROCK_COUNT} rocks — '
            f'Log topics /scout_fast_reasoning, /scout_slow_reasoning')

    # -----------------------------------------------------------------------
    # Terrain generation
    # -----------------------------------------------------------------------

    def _build_terrain_heights(self):
        """Load terrain from Depth Anything heightmap (falls back to flat)."""
        n = self._grid_size
        try:
            from .world_init import compute_heightmap, default_image_path
            image_path = os.environ.get('WILDFIRE_IMAGE_PATH') or default_image_path()
            h = compute_heightmap(image_path, n)
            self.get_logger().info(
                f'[SCENE] Terrain loaded from Depth Anything: '
                f'range=[{h.min():.2f}, {h.max():.2f}]')
            return h
        except Exception as exc:
            self.get_logger().warning(
                f'[SCENE] Depth Anything failed ({exc}); using flat terrain')
            return np.zeros((n, n), dtype=np.float64)

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

    def _terrain_z(self, wx: float, wy: float) -> float:
        """Look up terrain height at world coordinates."""
        n = self._grid_size
        cs = self._cell_size
        half = self._half_world
        gx = int((wx + half) / cs)
        gy = int((wy + half) / cs)
        if 0 <= gx < n and 0 <= gy < n:
            return float(self._terrain_h[gy, gx])
        return 0.0

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

    def _scout_reasoning_ros_cb(self, msg: String):
        """Fast loop: mirror streaming reasoning as foxglove.Log (same WS as /scene)."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = {'text': msg.data, 'parse_error': True}
        text = (data.get('text') or '').strip()
        parts = []
        if data.get('tick_id') is not None:
            parts.append(f"tick_id={data['tick_id']}")
        if data.get('streaming'):
            parts.append('streaming')
        us = data.get('updated_sec')
        if us is not None:
            try:
                parts.append(f'updated_sec={float(us):.3f}')
            except (TypeError, ValueError):
                parts.append(f'updated_sec={us!r}')
        header = f"[{' | '.join(parts)}]\n" if parts else ''
        foxglove.log(
            '/scout_fast_reasoning',
            _fg_scout_log(header + text, 'scout_fast_reasoning'),
        )

    def _scout_decision_ros_cb(self, msg: String):
        """Slow loop: tactical snapshot as foxglove.Log (JSON body)."""
        try:
            data = json.loads(msg.data)
            body = json.dumps(data, indent=2)
        except json.JSONDecodeError:
            body = msg.data
        foxglove.log(
            '/scout_slow_reasoning',
            _fg_scout_log(body, 'scout_slow_reasoning'),
        )

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

                # Red-dominant fire (avoid green at low intensity — matches bird's-eye viz)
                fr = min(1.0, 0.38 + 0.62 * (val ** 0.55))
                fg = min(0.88, 0.06 + 0.28 * val + 0.42 * val * val)
                fb = min(0.22, 0.04 + 0.10 * val * val * val)
                fire_cubes.append(fgs.CubePrimitive(
                    pose=_fg_pose(wx, wy, gz + fire_h / 2.0),
                    size=_fg_vec3(cs * 0.9, cs * 0.9, fire_h),
                    color=_fg_color(fr, fg, fb, 0.65 + 0.30 * val),
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
            tr = min(1.0, 0.25 + 0.75 * val)
            tg = min(0.7, 0.12 * val + 0.45 * val * val)
            tb = 0.04 * val
            burn_spheres.append(fgs.SpherePrimitive(
                pose=_fg_pose(wx, wy, gz + trunk_h + canopy_r * 0.6),
                size=_fg_vec3(canopy_r * 2.2, canopy_r * 2.2, canopy_r * 1.8),
                color=_fg_color(tr, tg, tb, 0.8),
            ))

        fire_entity = fgs.SceneEntity(
            timestamp=ts, frame_id=FRAME_ID, id='fire',
            lifetime=fgs.Duration(sec=2, nsec=0),
            cubes=fire_cubes, spheres=burn_spheres,
        )
        foxglove.log('/scene', fgs.SceneUpdate(entities=[fire_entity]))

    # -----------------------------------------------------------------------
    # Scout FOV perimeter
    # -----------------------------------------------------------------------

    def _build_fov_perimeter_cubes(self):
        """Return cube primitives for grid cells on the scout FOV circle edge."""
        sx, sy, sz = self._scout_pos
        sz = max(sz, 5.0)
        fov_radius = sz * math.tan(FOV_HALF_ANGLE_RAD)

        n = self._grid_size
        cs = self._cell_size
        half_cs = cs * 0.5

        cubes = []
        for gy in range(n):
            for gx in range(n):
                wx = (gx - n / 2.0 + 0.5) * cs
                wy = (gy - n / 2.0 + 0.5) * cs
                dist = math.hypot(wx - sx, wy - sy)
                if fov_radius - half_cs <= dist <= fov_radius + half_cs:
                    gz = float(self._terrain_h[gy, gx])
                    cubes.append(fgs.CubePrimitive(
                        pose=_fg_pose(wx, wy, gz + 0.06),
                        size=_fg_vec3(cs * 0.98, cs * 0.98, 0.12),
                        color=_fg_color(0.2, 0.4, 1.0, 0.5),
                    ))
        return cubes

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

        # firefighters — tall rectangular blocks (0.4 x 0.4 x 0.8) on terrain
        ff_w, ff_d, ff_h = 0.4, 0.4, 0.8
        for fid, (fx, fy, _fz) in self._ff_pos.items():
            gz = self._terrain_z(fx, fy)
            water_pct = self._ff_water.get(fid, 100.0)
            color = _fg_color(1.0, 0.85, 0.0) if water_pct > 20 else _fg_color(1.0, 0.4, 0.1)
            cubes.append(fgs.CubePrimitive(
                pose=_fg_pose(fx, fy, gz + ff_h / 2.0),
                size=_fg_vec3(ff_w, ff_d, ff_h),
                color=color,
            ))
            texts.append(fgs.TextPrimitive(
                pose=_fg_pose(fx, fy, gz + ff_h + 0.3),
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

        # Scout FOV perimeter — blue translucent tiles on the ring edge
        fov_cubes = self._build_fov_perimeter_cubes()
        fov_entity = fgs.SceneEntity(
            timestamp=ts, frame_id=FRAME_ID, id='fov_perimeter',
            lifetime=fgs.Duration(sec=1, nsec=0),
            cubes=fov_cubes,
        )

        foxglove.log('/scene', fgs.SceneUpdate(entities=[robot_entity, fov_entity]))


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
