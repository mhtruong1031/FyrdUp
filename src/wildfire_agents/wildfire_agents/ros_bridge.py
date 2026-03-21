#!/usr/bin/env python3
"""
Bridge between ROS 2 topics and agents.

The scout is now a Google ADK agent (ScoutADKAgent).  Firefighter uAgents
are kept for their state-machine logic but the scout communicates with them
through the bridge's ROS publishers rather than uAgent messages.
"""

import asyncio
import math
import os
import threading

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


class ROSBridge(Node):

    def __init__(self, scout_agent, firefighter_agents):
        super().__init__('ros_bridge')

        self.scout_agent = scout_agent
        self.firefighter_agents = firefighter_agents
        self.bridge = CvBridge()

        self._fire_grid_count = 0
        self._camera_count = 0
        self._odom_counts = {}
        self._analysis_running = False
        self._prev_ff_states = {}
        self._ff_fire_targets = {}

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

        # Periodic ADK analysis timer
        interval = scout_agent.analysis_interval
        self.create_timer(interval, self._trigger_analysis)

        self.get_logger().info(
            f'[BRIDGE] Ready — ADK scout, '
            f'firefighters={list(firefighter_agents.keys())}')

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
            if new_state == 'FIGHTING':
                target = self._ff_fire_targets.get(ff_id)
                if target is not None:
                    dist = math.hypot(p.x - target[0], p.y - target[1])
                    if dist <= SPRAY_RANGE:
                        self.spray_enable_pubs[ff_id].publish(Bool(data=True))
                    else:
                        self.spray_enable_pubs[ff_id].publish(Bool(data=False))
                else:
                    self.spray_enable_pubs[ff_id].publish(Bool(data=False))
            elif prev_state == 'FIGHTING':
                self.spray_enable_pubs[ff_id].publish(Bool(data=False))

        self.scout_agent.update_firefighter(
            ff_id, (p.x, p.y), ff.water_level, ff.state)

    def _water_cb(self, ff_id, msg):
        if ff_id in self.firefighter_agents:
            self.firefighter_agents[ff_id].update_water_level(msg.data)
            ff = self.firefighter_agents[ff_id]
            self.scout_agent.update_firefighter(
                ff_id, ff.position, msg.data, ff.state)
        else:
            self.get_logger().warning(
                f'[BRIDGE] Water level for unknown {ff_id}')

    # -- ADK scout callbacks -------------------------------------------------

    def _handle_scout_move(self, x, y, z):
        pt = Point(x=float(x), y=float(y), z=float(z))
        self.scout_target_pub.publish(pt)
        self.get_logger().info(
            f'[BRIDGE] Scout move → ({x:.1f}, {y:.1f}, {z:.1f})')

    def _handle_assign_firefighter(self, ff_id, target_x, target_y):
        pt = Point(x=float(target_x), y=float(target_y), z=0.0)
        self._ff_fire_targets[ff_id] = (target_x, target_y)
        if ff_id in self.target_position_pubs:
            self.target_position_pubs[ff_id].publish(pt)
            self.get_logger().info(
                f'[BRIDGE] Assign {ff_id} → ({target_x:.1f}, {target_y:.1f})')
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
            self.get_logger().info('[BRIDGE] ADK analysis still running — skipping')
            return
        self._analysis_running = True
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _run_analysis_thread(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply(loop)
            loop.run_until_complete(
                asyncio.wait_for(self.scout_agent.run_analysis(), timeout=60.0)
            )
        except asyncio.TimeoutError:
            print('[BRIDGE] ADK analysis timed out (60s)')
        except Exception as exc:
            print(f'[BRIDGE] ADK analysis error: {exc}')
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

    rclpy.init(args=args)

    num_firefighters = int(os.environ.get('NUM_FIREFIGHTERS', '4'))
    use_vlm = os.environ.get('USE_VLM', 'true').lower() == 'true'

    print(f'[BRIDGE] Starting with NUM_FIREFIGHTERS={num_firefighters} '
          f'USE_VLM={use_vlm}')

    # ADK scout (no uAgent, no port)
    scout_agent = ScoutADKAgent(use_vlm=use_vlm)
    print(f'[BRIDGE] ADK Scout agent created')

    # Firefighter uAgents (still use Fetch uAgents for state machine)
    # Use a dummy scout address since ADK scout has no uAgent address
    dummy_scout_addr = 'agent1qfplaceholder000000000000000000000000000000000000000000000'

    firefighter_agents = {}
    for i in range(num_firefighters):
        ff_id = f'firefighter_{i + 1}'
        ff = FirefighterAgent(
            firefighter_id=ff_id,
            port=8001 + i,
            scout_address=dummy_scout_addr,
        )
        firefighter_agents[ff_id] = ff
        print(f'[BRIDGE] Firefighter {ff_id} created: port={8001 + i}')

    bridge = ROSBridge(scout_agent, firefighter_agents)

    # Start firefighter uAgent threads (they still manage state machines)
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
