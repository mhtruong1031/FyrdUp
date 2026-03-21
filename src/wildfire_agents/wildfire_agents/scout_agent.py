#!/usr/bin/env python3
"""
Scout uAgent — aerial coordinator for wildfire firefighting.

Periodically analyses the fire via a Gemini VLM (or grid fallback) and
sends tactical MoveCommand / RefillCommand messages to firefighter agents.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
from uagents import Agent, Context, Model

from .vlm_gemini import GeminiVLM


# ---------------------------------------------------------------------------
# uAgent message models (shared with firefighter_agent)
# ---------------------------------------------------------------------------

class FireMap(Model):
    fire_locations: List[Tuple[int, int]]
    fire_intensity: List[float]
    wind_direction: float
    threat_level: str


class MoveCommand(Model):
    target_id: str
    position: Tuple[float, float]
    priority: int
    fire_cell: Tuple[int, int]


class RefillCommand(Model):
    target_id: str
    refill_position: Tuple[float, float]


class StatusUpdate(Model):
    firefighter_id: str
    position: Tuple[float, float]
    water_level: float
    state: str  # IDLE, MOVING, FIGHTING, REFILLING


# ---------------------------------------------------------------------------
# Scout agent
# ---------------------------------------------------------------------------

class ScoutAgent:

    def __init__(self, port: int = 8000, use_vlm: bool = True,
                 analysis_interval: float = 10.0):
        self.agent = Agent(name='scout', port=port, seed='scout_seed')
        self.use_vlm = use_vlm
        self.analysis_interval = analysis_interval

        if use_vlm:
            try:
                self.vlm = GeminiVLM()
                print('Scout: Gemini VLM initialized')
            except Exception as e:
                print(f'Scout: Failed to initialize VLM: {e}')
                print('Scout: Running without VLM (grid-based fallback)')
                self.use_vlm = False

        self.firefighter_addresses: Dict[str, str] = {}
        self.firefighter_status: Dict[str, dict] = {}

        self.latest_image = None
        self.fire_grid = None
        self.grid_size = 50

        self.fire_map = None

        self._setup_handlers()

    # -- handlers ------------------------------------------------------------

    def _setup_handlers(self):

        @self.agent.on_interval(period=self.analysis_interval)
        async def analyze_and_command(ctx: Context):
            ctx.logger.info(
                f'[SCOUT] === Analysis tick === '
                f'registered={list(self.firefighter_addresses.keys())} '
                f'status_count={len(self.firefighter_status)} '
                f'mode={"VLM" if self.use_vlm else "grid"}')

            if self.use_vlm:
                if self.latest_image is None:
                    ctx.logger.warning('[SCOUT] No camera image available yet — skipping')
                    return
                ctx.logger.info(f'[SCOUT] Running VLM analysis on image...')
                fire_analysis = await self.vlm.analyze_fire(self.latest_image)
            else:
                if self.fire_grid is None:
                    ctx.logger.warning('[SCOUT] No fire grid data yet — skipping')
                    return
                ctx.logger.info('[SCOUT] Running grid-based analysis...')
                fire_analysis = self._analyze_fire_grid()

            n_fires = len(fire_analysis.get('fire_locations', []))
            n_recs = len(fire_analysis.get('recommended_positions', []))
            ctx.logger.info(
                f'[SCOUT] Analysis result: '
                f'fires={n_fires} recommended_pos={n_recs} '
                f'threat={fire_analysis.get("threat_level", "?")} '
                f'| {fire_analysis.get("analysis", "")}')

            if n_recs == 0:
                ctx.logger.warning(
                    '[SCOUT] No recommended positions from analysis — '
                    'no commands will be sent')

            self.fire_map = FireMap(
                fire_locations=fire_analysis['fire_locations'],
                fire_intensity=fire_analysis['fire_intensity'],
                wind_direction=fire_analysis['wind_direction'],
                threat_level=fire_analysis['threat_level'],
            )

            if not self.firefighter_status:
                ctx.logger.warning(
                    '[SCOUT] firefighter_status is EMPTY — '
                    'no firefighters have reported in. '
                    'Cannot allocate. Are StatusUpdates arriving?')

            commands = self._allocate_firefighters(
                fire_analysis['recommended_positions'], ctx)

            ctx.logger.info(
                f'[SCOUT] Allocation produced {len(commands)} commands '
                f'for {len(self.firefighter_status)} known firefighters')

            for cmd in commands:
                if cmd.target_id in self.firefighter_addresses:
                    addr = self.firefighter_addresses[cmd.target_id]
                    await ctx.send(addr, cmd)
                    ctx.logger.info(
                        f'[SCOUT] Sent MoveCommand to {cmd.target_id} '
                        f'addr={addr[:20]}... '
                        f'pos=({cmd.position[0]:.1f}, {cmd.position[1]:.1f}) '
                        f'priority={cmd.priority} '
                        f'fire_cell={cmd.fire_cell}')
                else:
                    ctx.logger.warning(
                        f'[SCOUT] Cannot send to {cmd.target_id} — '
                        f'no address registered! '
                        f'Known addresses: {list(self.firefighter_addresses.keys())}')

        @self.agent.on_message(model=StatusUpdate)
        async def handle_status(ctx: Context, sender: str, msg: StatusUpdate):
            ctx.logger.info(
                f'[SCOUT] StatusUpdate from {msg.firefighter_id}: '
                f'state={msg.state} water={msg.water_level:.0f}% '
                f'pos=({msg.position[0]:.1f}, {msg.position[1]:.1f}) '
                f'sender={sender[:20]}...')

            self.firefighter_status[msg.firefighter_id] = {
                'position': msg.position,
                'water_level': msg.water_level,
                'state': msg.state,
                'address': sender,
            }

            if msg.firefighter_id not in self.firefighter_addresses:
                self.firefighter_addresses[msg.firefighter_id] = sender
                ctx.logger.info(
                    f'[SCOUT] NEW firefighter registered: {msg.firefighter_id} '
                    f'addr={sender[:20]}... '
                    f'total_registered={len(self.firefighter_addresses)}')

            if msg.water_level < 20.0 and msg.state != 'REFILLING':
                refill_cmd = RefillCommand(
                    target_id=msg.firefighter_id,
                    refill_position=(0.0, 0.0),
                )
                await ctx.send(sender, refill_cmd)
                ctx.logger.info(
                    f'[SCOUT] Sent RefillCommand to {msg.firefighter_id} '
                    f'(water={msg.water_level:.1f}%)')

    # -- allocation ----------------------------------------------------------

    def _allocate_firefighters(self, recommended_positions: List,
                               ctx: Context = None) -> List[MoveCommand]:
        commands = []
        _log = ctx.logger.info if ctx else (lambda m: None)
        _warn = ctx.logger.warning if ctx else (lambda m: None)

        available = []
        for ff_id, status in self.firefighter_status.items():
            if status['state'] in ('IDLE', 'MOVING') and status['water_level'] > 20.0:
                available.append((ff_id, status['position']))
                _log(f'[SCOUT] Allocate: {ff_id} AVAILABLE '
                     f'state={status["state"]} water={status["water_level"]:.0f}%')
            else:
                _log(f'[SCOUT] Allocate: {ff_id} SKIPPED '
                     f'state={status["state"]} water={status["water_level"]:.0f}%')

        if not available:
            _warn('[SCOUT] Allocate: NO available firefighters '
                  '(need state IDLE/MOVING and water > 20%)')
        if not recommended_positions:
            _warn('[SCOUT] Allocate: NO recommended positions to assign')

        _log(f'[SCOUT] Allocate: {len(available)} available, '
             f'{len(recommended_positions)} positions to fill')

        assigned_positions = set()
        for ff_id, ff_pos in available:
            if not recommended_positions:
                break

            min_dist = float('inf')
            best_pos = None
            best_idx = None

            for idx, pos in enumerate(recommended_positions):
                if idx in assigned_positions:
                    continue
                world_pos = self._grid_to_world(pos)
                dist = np.sqrt(
                    (world_pos[0] - ff_pos[0]) ** 2 +
                    (world_pos[1] - ff_pos[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = world_pos
                    best_idx = idx

            if best_pos is not None:
                assigned_positions.add(best_idx)
                commands.append(MoveCommand(
                    target_id=ff_id,
                    position=best_pos,
                    priority=100 - int(min_dist * 10),
                    fire_cell=recommended_positions[best_idx],
                ))
                _log(f'[SCOUT] Allocate: ASSIGNED {ff_id} -> '
                     f'grid={recommended_positions[best_idx]} '
                     f'world=({best_pos[0]:.1f}, {best_pos[1]:.1f}) '
                     f'dist={min_dist:.1f}')

        return commands

    def _grid_to_world(self, grid_coords) -> Tuple[float, float]:
        half = self.grid_size / 2.0
        x = grid_coords[0] - half + 0.5
        y = grid_coords[1] - half + 0.5
        return (x, y)

    # -- grid-based fallback analysis ----------------------------------------

    def _analyze_fire_grid(self) -> Dict:
        n = self.grid_size
        grid = self.fire_grid

        fire_locations = []
        fire_intensity = []
        for y in range(n):
            for x in range(n):
                val = grid[y * n + x]
                if val > 0.1:
                    fire_locations.append([x, y])
                    fire_intensity.append(val)

        # Recommend perimeter cells adjacent to the fire
        burning = set(map(tuple, fire_locations))
        recommended = []
        seen = set()
        for fx, fy in fire_locations:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in burning:
                    if (nx, ny) not in seen:
                        seen.add((nx, ny))
                        recommended.append([nx, ny])
        recommended = recommended[:8]

        num_burning = len(fire_locations)
        threat = ('low' if num_burning < 10
                  else ('medium' if num_burning < 30 else 'high'))

        return {
            'fire_locations': fire_locations,
            'fire_intensity': fire_intensity,
            'wind_direction': 45.0,
            'recommended_positions': recommended,
            'threat_level': threat,
            'analysis': (
                f'{num_burning} cells burning, threat {threat}. '
                f'Recommending {len(recommended)} perimeter positions.'),
        }

    # -- data setters (called by ros_bridge) ---------------------------------

    def set_camera_image(self, image: np.ndarray):
        self.latest_image = image

    def set_fire_grid(self, intensity: list, grid_size: int):
        self.fire_grid = intensity
        self.grid_size = grid_size

    def run(self):
        self.agent.run()


def main():
    use_vlm = os.environ.get('USE_VLM', 'true').lower() == 'true'
    scout = ScoutAgent(use_vlm=use_vlm)
    scout.run()


if __name__ == '__main__':
    main()
