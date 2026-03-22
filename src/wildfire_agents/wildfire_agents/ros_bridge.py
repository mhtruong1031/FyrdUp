#!/usr/bin/env python3
"""
Bridge between ROS 2 topics and agents.

The scout is a Google ADK agent (ScoutADKAgent) wrapped by a thin uAgent
(ScoutUAgent) for messaging.  Commands flow to firefighter uAgents via the
scout uAgent protocol; ROS topics drive the Gazebo simulation.
"""

import asyncio
import logging
import math
import os
import threading
import time

import nest_asyncio
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from wildfire_msgs.msg import FireGrid
from cv_bridge import CvBridge


SPRAY_RANGE = 3.0  # metres — must match fire_grid_node spray_range
FIRE_THRESHOLD = 0.1
# Re-ping scout via uAgent while still on a burning cell (seconds)
IN_FIRE_UAGENT_PING_COOLDOWN_S = 12.0


class ROSBridge(Node):

    def __init__(self, scout_agent, firefighter_agents, scout_uagent):
        super().__init__('ros_bridge')

        self.scout_agent = scout_agent
        self.firefighter_agents = firefighter_agents
        self.scout_uagent = scout_uagent
        self.bridge = CvBridge()

        self._fire_grid_count = 0
        self._camera_count = 0
        self._odom_counts = {}
        self._analysis_running = False
        self._prev_ff_states = {}
        self._ff_fire_targets = {}
        self._prev_water_levels: dict[str, float] = {}
        self._last_fire_grid = None  # last FireGrid msg for spray proximity
        self._ff_last_spray: dict[str, bool] = {}
        self._ff_on_burning_cell_prev: dict[str, bool] = {}
        self._ff_last_in_fire_ping_mono: dict[str, float] = {}

        # Bird's-eye image from viz_renderer (kept for compatibility)
        self.create_subscription(
            Image, '/viz/fire_image', self._camera_cb, 10)

        # Fire grid → scout
        self.create_subscription(
            FireGrid, '/fire_grid', self._fire_grid_cb, 10)

        # Scout odometry → track position for ADK agent
        self.create_subscription(
            Odometry, '/scout/odometry', self._scout_odom_cb, 10)

        # Scout target position publisher (ADK agent's move_scout tool)
        self.scout_target_pub = self.create_publisher(
            Point, '/scout/target_position', 10)

        # Per-firefighter subscriptions and publishers
        self.target_position_pubs = {}
        self.spray_enable_pubs = {}

        for ff_id, ff_agent in firefighter_agents.items():
            self._odom_counts[ff_id] = 0

            self.create_subscription(
                Odometry, f'/{ff_id}/odometry',
                lambda msg, fid=ff_id: self._odom_cb(fid, msg), 10)

            self.create_subscription(
                Float32, f'/{ff_id}/water_level',
                lambda msg, fid=ff_id: self._water_cb(fid, msg), 10)

            ff_agent.on_move_command = (
                lambda pos, cell, fid=ff_id: self._send_move(fid, pos, cell))
            ff_agent.on_refill_command = (
                lambda pos, fid=ff_id: self._send_refill(fid, pos))

            self.target_position_pubs[ff_id] = self.create_publisher(
                Point, f'/{ff_id}/target_position', 10)
            self.spray_enable_pubs[ff_id] = self.create_publisher(
                Bool, f'/{ff_id}/spray_enable', 10)

            self.get_logger().info(
                f'[BRIDGE] Wired {ff_id}: '
                f'topics=/{ff_id}/target_position, /{ff_id}/spray_enable')

        # Wire ADK scout callbacks
        self.scout_agent.on_move_scout = self._handle_scout_move
        self.scout_agent.on_assign_firefighter = self._handle_assign_firefighter
        self.scout_agent.on_refill_firefighter = self._handle_refill_firefighter
        scout_uagent.on_in_fire_alert = self._on_in_fire_uagent_alert

        # Periodic ADK analysis timer
        interval = scout_agent.analysis_interval
        self.create_timer(interval, self._trigger_analysis)

        self.get_logger().info(
            f'[BRIDGE] Ready — ADK scout, '
            f'firefighters={list(firefighter_agents.keys())}')

    def _world_to_grid(self, wx, wy, width, cell_size):
        half = width * cell_size / 2.0
        gx = int((wx + half) / cell_size)
        gy = int((wy + half) / cell_size)
        return max(0, min(gx, width - 1)), max(0, min(gy, width - 1))

    def _fire_within_range(self, rx, ry, range_m: float) -> bool:
        msg = self._last_fire_grid
        if msg is None or not msg.intensity:
            return False
        n = msg.width
        cs = msg.cell_size
        r_cells = int(range_m / cs) + 2
        gx0, gy0 = self._world_to_grid(rx, ry, n, cs)
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                gx, gy = gx0 + dx, gy0 + dy
                if 0 <= gx < n and 0 <= gy < n:
                    if msg.intensity[gy * n + gx] >= FIRE_THRESHOLD:
                        wx = (gx - n / 2.0 + 0.5) * cs
                        wy = (gy - n / 2.0 + 0.5) * cs
                        if math.hypot(wx - rx, wy - ry) <= range_m:
                            return True
        return False

    def _position_on_burning_cell(self, wx: float, wy: float) -> bool:
        """True if the robot's grid cell (footprint center) is burning."""
        msg = self._last_fire_grid
        if msg is None or not msg.intensity:
            return False
        n = msg.width
        gx, gy = self._world_to_grid(wx, wy, n, msg.cell_size)
        idx = gy * n + gx
        if 0 <= idx < len(msg.intensity):
            return msg.intensity[idx] >= FIRE_THRESHOLD
        return False

    def _on_in_fire_uagent_alert(self, msg):
        self.get_logger().warn(
            f'[BRIDGE] InFireAlert from {msg.firefighter_id} ({msg.reason!r}) '
            f'pos=({msg.position[0]:.1f},{msg.position[1]:.1f}) — triggering scout')
        self._trigger_analysis()

    def _update_spray(self, ff_id: str, x: float, y: float, ff):
        """Spray only when within range of actual fire (from grid), including while MOVING."""
        if ff.state == 'REFILLING' or ff.water_level <= 0.5:
            want = False
        elif ff.state in ('MOVING', 'FIGHTING'):
            want = self._fire_within_range(x, y, SPRAY_RANGE)
        else:
            want = False
        prev = self._ff_last_spray.get(ff_id)
        if prev != want:
            self._ff_last_spray[ff_id] = want
            if ff_id in self.spray_enable_pubs:
                self.spray_enable_pubs[ff_id].publish(Bool(data=want))

    # -- inbound callbacks ---------------------------------------------------

    def _camera_cb(self, msg):
        self._camera_count += 1
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.scout_agent.set_camera_image(cv_image)
        except Exception as e:
            self.get_logger().error(f'[BRIDGE] Camera conversion error: {e}')

    def _fire_grid_cb(self, msg: FireGrid):
        self._fire_grid_count += 1
        self._last_fire_grid = msg
        self.scout_agent.set_fire_grid(list(msg.intensity), msg.width)
        if self._fire_grid_count % 10 == 1:
            burning = sum(1 for v in msg.intensity if v > 0.1)
            self.get_logger().info(
                f'[BRIDGE] FireGrid #{self._fire_grid_count} received '
                f'size={msg.width}x{msg.height} burning_cells={burning}')

    def _scout_odom_cb(self, msg):
        p = msg.pose.pose.position
        self.scout_agent.update_scout_position(p.x, p.y, p.z)

    def _odom_cb(self, ff_id, msg):
        p = msg.pose.pose.position
        self._odom_counts[ff_id] = self._odom_counts.get(ff_id, 0) + 1
        if ff_id not in self.firefighter_agents:
            self.get_logger().warning(f'[BRIDGE] Odom for unknown {ff_id}')
            return

        ff = self.firefighter_agents[ff_id]
        prev_state = self._prev_ff_states.get(ff_id, ff.state)
        ff.update_position(p.x, p.y)
        new_state = ff.state

        if new_state != prev_state:
            self._prev_ff_states[ff_id] = new_state

        on_fire = self._position_on_burning_cell(p.x, p.y)
        was_on_fire = self._ff_on_burning_cell_prev.get(ff_id, False)
        if on_fire:
            now = time.monotonic()
            last_ping = self._ff_last_in_fire_ping_mono.get(ff_id, 0.0)
            if (not was_on_fire) or (
                now - last_ping >= IN_FIRE_UAGENT_PING_COOLDOWN_S
            ):
                ff.enqueue_in_fire_alert("on_burning_cell")
                self._ff_last_in_fire_ping_mono[ff_id] = now
                if not was_on_fire:
                    self.get_logger().warn(
                        f'[BRIDGE] {ff_id} entered burning cell — InFireAlert → scout')
        self._ff_on_burning_cell_prev[ff_id] = on_fire

        self._update_spray(ff_id, p.x, p.y, ff)

        self.scout_agent.update_firefighter(
            ff_id, (p.x, p.y), ff.water_level, ff.state)

    def _water_cb(self, ff_id, msg):
        if ff_id not in self.firefighter_agents:
            self.get_logger().warning(
                f'[BRIDGE] Water level for unknown {ff_id}')
            return

        prev = self._prev_water_levels.get(ff_id, 100.0)
        cur = msg.data
        self._prev_water_levels[ff_id] = cur

        self.firefighter_agents[ff_id].update_water_level(cur)
        ff = self.firefighter_agents[ff_id]
        self.scout_agent.update_firefighter(
            ff_id, ff.position, cur, ff.state)

        px, py = ff.position
        self._update_spray(ff_id, px, py, ff)

        # Detect downward crossings of the 20% and 0% thresholds
        TANK_CAPACITY = 100.0
        threshold_20 = TANK_CAPACITY * 0.20
        crossed_20 = prev > threshold_20 and cur <= threshold_20
        crossed_0 = prev > 0.0 and cur <= 0.0

        if crossed_20 or crossed_0:
            pct = cur / TANK_CAPACITY * 100.0
            tag = '0%' if crossed_0 else '20%'
            self.get_logger().warn(
                f'[BRIDGE] WATER ALERT: {ff_id} hit {tag} '
                f'({cur:.1f}/{TANK_CAPACITY:.0f}) — pinging scout')
            self._trigger_analysis()

    # -- ADK scout callbacks -------------------------------------------------

    def _handle_scout_move(self, x, y, z):
        pt = Point(x=float(x), y=float(y), z=float(z))
        self.scout_target_pub.publish(pt)
        self.get_logger().info(
            f'[BRIDGE] Scout move → ({x:.1f}, {y:.1f}, {z:.1f})')

    def _handle_assign_firefighter(self, ff_id, target_x, target_y):
        pt = Point(x=float(target_x), y=float(target_y), z=0.0)
        self._ff_fire_targets[ff_id] = (target_x, target_y)

        # Send MoveCommand via the scout uAgent instead of mutating ff directly
        n = self.scout_agent.grid_size
        cs = self.scout_agent.cell_size
        half = n * cs / 2.0
        fire_cell = (
            int((target_x + half) / cs),
            int((target_y + half) / cs),
        )
        self.scout_uagent.send_move_command(
            ff_id, (target_x, target_y), fire_cell)

        if ff_id in self.target_position_pubs:
            self.target_position_pubs[ff_id].publish(pt)
            self.get_logger().info(
                f'[BRIDGE] Assign {ff_id} → ({target_x:.1f}, {target_y:.1f})')
        else:
            self.get_logger().error(
                f'[BRIDGE] No publisher for {ff_id}')

    def _handle_refill_firefighter(self, ff_id):
        refill_pos = (0.0, 0.0)
        pt = Point(x=refill_pos[0], y=refill_pos[1], z=0.0)

        # Send RefillCommand via the scout uAgent instead of mutating ff directly
        self.scout_uagent.send_refill_command(ff_id, refill_pos)

        if ff_id in self.spray_enable_pubs:
            self.spray_enable_pubs[ff_id].publish(Bool(data=False))
        if ff_id in self.target_position_pubs:
            self.target_position_pubs[ff_id].publish(pt)
            self.get_logger().info(
                f'[BRIDGE] Refill {ff_id} → ({refill_pos[0]:.1f}, {refill_pos[1]:.1f})')
        else:
            self.get_logger().error(
                f'[BRIDGE] No publisher for {ff_id}')

    # -- outbound commands (from firefighter uAgents) ------------------------

    def _send_move(self, ff_id, position, fire_cell):
        pt = Point(x=float(position[0]), y=float(position[1]), z=0.0)
        self._ff_fire_targets[ff_id] = (float(position[0]), float(position[1]))
        if ff_id in self.target_position_pubs:
            self.target_position_pubs[ff_id].publish(pt)
            self.get_logger().info(
                f'[BRIDGE] PUBLISH /{ff_id}/target_position '
                f'({position[0]:.1f}, {position[1]:.1f}) '
                f'fire_cell={fire_cell}')
        else:
            self.get_logger().error(
                f'[BRIDGE] No publisher for {ff_id} — cannot send move!')

    def _send_refill(self, ff_id, position):
        if ff_id in self.spray_enable_pubs:
            self.spray_enable_pubs[ff_id].publish(Bool(data=False))
        pt = Point(x=float(position[0]), y=float(position[1]), z=0.0)
        if ff_id in self.target_position_pubs:
            self.target_position_pubs[ff_id].publish(pt)
            self.get_logger().info(
                f'[BRIDGE] PUBLISH /{ff_id}/target_position '
                f'({position[0]:.1f}, {position[1]:.1f}) refill + spray=OFF')

    # -- ADK analysis trigger ------------------------------------------------

    def _trigger_analysis(self):
        if self._analysis_running:
            self.get_logger().info('[BRIDGE] Scout analysis still running — skipping')
            return
        self._analysis_running = True
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _run_analysis_thread(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply(loop)
            loop.run_until_complete(
                asyncio.wait_for(self.scout_agent.run_analysis(), timeout=30.0)
            )
        except asyncio.TimeoutError:
            print('[BRIDGE] Scout analysis timed out (30s)')
        except Exception as exc:
            print(f'[BRIDGE] Scout analysis error: {exc}')
        finally:
            self._analysis_running = False


def _run_agent(agent):
    """Run a uAgent in a fresh event loop on its own daemon thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)
    agent.run()


def main(args=None):
    from .scout_agent import ScoutADKAgent
    from .firefighter_agent import FirefighterAgent
    from .scout_uagent import ScoutUAgent

    logging.getLogger('uagents').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    rclpy.init(args=args)

    num_firefighters = int(os.environ.get('NUM_FIREFIGHTERS', '4'))
    use_vlm = os.environ.get('USE_VLM', 'true').lower() == 'true'

    print(f'[BRIDGE] Starting with NUM_FIREFIGHTERS={num_firefighters} '
          f'USE_VLM={use_vlm}')

    # ADK scout (no uAgent, no port)
    scout_agent = ScoutADKAgent(use_vlm=use_vlm)
    print(f'[BRIDGE] ADK Scout agent created')

    # Scout uAgent wrapper (messaging layer around the ADK agent)
    scout_uagent = ScoutUAgent(adk_agent=scout_agent)
    print(f'[BRIDGE] Scout uAgent created: address={scout_uagent.address}')

    # Firefighter uAgents
    firefighter_agents = {}
    for i in range(num_firefighters):
        ff_id = f'firefighter_{i + 1}'
        ff = FirefighterAgent(
            firefighter_id=ff_id,
            port=8001 + i,
            scout_address=scout_uagent.address,
        )
        scout_uagent.register_firefighter(ff_id, ff.agent.address)
        firefighter_agents[ff_id] = ff
        print(f'[BRIDGE] Firefighter {ff_id} created: port={8001 + i}')

    bridge = ROSBridge(scout_agent, firefighter_agents, scout_uagent)

    # Start scout uAgent thread
    threading.Thread(
        target=_run_agent, args=(scout_uagent,), daemon=True).start()
    print('[BRIDGE] Scout uAgent thread started')

    # Start firefighter uAgent threads
    print('[BRIDGE] Starting firefighter uAgent threads...')
    for ff_id, ff in firefighter_agents.items():
        threading.Thread(
            target=_run_agent, args=(ff,), daemon=True).start()
        print(f'[BRIDGE] {ff_id} uAgent thread started')

    print('[BRIDGE] All agents running. Spinning ROS bridge...')

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
