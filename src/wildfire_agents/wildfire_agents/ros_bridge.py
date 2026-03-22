#!/usr/bin/env python3
"""
Bridge between ROS 2 topics and agents.

The scout is a Google ADK agent (ScoutADKAgent) wrapped by a thin uAgent
(ScoutUAgent) for messaging.  Commands flow to firefighter uAgents via the
scout uAgent protocol; ROS topics drive the Gazebo simulation.
"""

import asyncio
import json
import logging
from typing import Optional
import math
import os
import threading
import time
from urllib.parse import quote

import nest_asyncio
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32, String
from nav_msgs.msg import Odometry
from wildfire_msgs.msg import FireGrid
from cv_bridge import CvBridge


SPRAY_RANGE = 3.0  # metres — must match fire_grid_node spray_range
FIRE_THRESHOLD = 0.1
# Re-ping scout via uAgent while still on a burning cell (seconds)
IN_FIRE_UAGENT_PING_COOLDOWN_S = 12.0


def _local_agent_inspector_url(agent) -> str:
    """
    Agentverse Local Agent Inspector URL for a running uAgent (same shape as
    framework logs). Requires uAgents 0.18+ so the local HTTP server exposes
    inspector routes; 0.11 only had /submit and the web UI could not attach.
    """
    av = agent.agentverse
    if isinstance(av, dict):
        base = f"{av['http_prefix']}://{av['base_url']}".rstrip('/')
    else:
        base = getattr(av, 'url', None) or 'https://agentverse.ai'
        base = str(base).rstrip('/')
    port = getattr(agent, '_port', None)
    if port is None:
        port = 8000
    uri = quote(f'http://127.0.0.1:{port}')
    return f'{base}/inspect/?uri={uri}&address={agent.address}'


def _save_inspector_links_file(path: str, scout_uagent, firefighter_agents: dict) -> None:
    """Write Local Agent Inspector URLs to a text file (one label + URL per line)."""
    lines = [
        '# Local Agent Inspector (uAgents) — open these URLs in your browser',
        '',
        f'scout: {_local_agent_inspector_url(scout_uagent.agent)}',
    ]
    for ff_id, ff in firefighter_agents.items():
        lines.append(f'{ff_id}: {_local_agent_inspector_url(ff.agent)}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


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

        self._reasoning_running = False
        self._last_reasoning_pub_payload: Optional[str] = None

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

        self.scout_reasoning_pub = self.create_publisher(
            String, '/scout/reasoning_status', 10)
        self.scout_decision_pub = self.create_publisher(
            String, '/scout/decision_snapshot', 10)

        self._reasoning_enabled = (
            os.environ.get('REASONING_ENABLED', 'true').lower() == 'true')

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
        self.scout_agent.on_decision_snapshot = self._publish_decision_snapshot
        scout_uagent.on_in_fire_alert = self._on_in_fire_uagent_alert

        # Periodic ADK analysis timer (slow loop)
        interval = scout_agent.analysis_interval
        self.create_timer(interval, self._trigger_analysis)

        if self._reasoning_enabled:
            r_int = max(1.0, float(scout_agent.reasoning_interval))
            self.create_timer(r_int, self._trigger_reasoning)
            self.get_logger().info(
                f'[BRIDGE] Fast reasoning loop every {r_int}s')

        # Throttled ROS mirror of reasoning buffer for Foxglove (2–4 Hz)
        self.create_timer(0.3, self._publish_reasoning_status_tick)

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

    # -- fast reasoning loop -------------------------------------------------

    def _trigger_reasoning(self):
        if not self._reasoning_enabled:
            return
        if self._reasoning_running:
            self.get_logger().debug(
                '[BRIDGE] Scout reasoning still running — skipping')
            return
        self._reasoning_running = True
        threading.Thread(
            target=self._run_reasoning_thread, daemon=True).start()

    def _run_reasoning_thread(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply(loop)
            loop.run_until_complete(
                asyncio.wait_for(
                    self.scout_agent.run_reasoning_stream(), timeout=90.0)
            )
        except asyncio.TimeoutError:
            print('[BRIDGE] Scout reasoning timed out (90s)')
        except Exception as exc:
            print(f'[BRIDGE] Scout reasoning error: {exc}')
        finally:
            self._reasoning_running = False

    def _publish_reasoning_status_tick(self):
        d = self.scout_agent.get_reasoning_status_dict()
        try:
            js = json.dumps(d, default=str)
        except (TypeError, ValueError):
            return
        streaming = d.get('streaming')
        if js == self._last_reasoning_pub_payload and not streaming:
            return
        self._last_reasoning_pub_payload = js
        self.scout_reasoning_pub.publish(String(data=js))

    def _publish_decision_snapshot(self, payload: dict):
        try:
            self.scout_decision_pub.publish(
                String(data=json.dumps(payload, default=str)))
        except (TypeError, ValueError) as exc:
            self.get_logger().error(
                f'[BRIDGE] decision snapshot JSON error: {exc}')


def _run_agent(agent):
    """Run a uAgent in a fresh event loop on its own daemon thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)
    agent.run()


def _load_agent_params() -> dict:
    """Load ``analysis_interval`` / ``reasoning_interval`` from package YAML."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('wildfire_agents')
        path = os.path.join(share, 'config', 'agent_config.yaml')
        with open(path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        ros = (cfg.get('ros_bridge') or {}).get('ros__parameters') or {}
        return {
            'analysis_interval': float(ros.get('analysis_interval', 20.0)),
            'reasoning_interval': float(ros.get('reasoning_interval', 5.0)),
        }
    except Exception as exc:
        print(f'[BRIDGE] agent_config.yaml not loaded ({exc!r}) — using defaults')
        return {
            'analysis_interval': 20.0,
            'reasoning_interval': 5.0,
        }


def main(args=None):
    from .scout_agent import ScoutADKAgent
    from .firefighter_agent import FirefighterAgent
    from .scout_uagent import ScoutUAgent

    logging.getLogger('uagents').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    rclpy.init(args=args)

    num_firefighters = int(os.environ.get('NUM_FIREFIGHTERS', '4'))
    use_vlm = os.environ.get('USE_VLM', 'true').lower() == 'true'

    params = _load_agent_params()
    analysis_interval = float(os.environ.get(
        'ANALYSIS_INTERVAL', params['analysis_interval']))
    reasoning_interval = float(os.environ.get(
        'REASONING_INTERVAL', params['reasoning_interval']))

    print(f'[BRIDGE] Starting with NUM_FIREFIGHTERS={num_firefighters} '
          f'USE_VLM={use_vlm}')

    # ADK scout (no uAgent, no port)
    scout_agent = ScoutADKAgent(
        use_vlm=use_vlm,
        analysis_interval=analysis_interval,
        reasoning_interval=reasoning_interval,
    )
    print(
        f'[BRIDGE] ADK Scout agent created '
        f'(analysis_interval={analysis_interval}s, '
        f'reasoning_interval={reasoning_interval}s)'
    )

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

    links_path = os.environ.get(
        'UAGENT_INSPECTOR_LINKS_FILE', 'uagent_inspector_links.txt')
    try:
        _save_inspector_links_file(links_path, scout_uagent, firefighter_agents)
        print(
            f'[BRIDGE] Inspector links saved to {os.path.abspath(links_path)}')
    except OSError as exc:
        print(f'[BRIDGE] Could not write inspector links file ({exc!r})')

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
