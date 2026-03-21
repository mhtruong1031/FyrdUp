#!/usr/bin/env python3
"""
Bridge between ROS 2 topics and Fetch uAgent instances.

Reads USE_VLM and NUM_FIREFIGHTERS from environment variables (set by the
launch file via SetEnvironmentVariable).
"""

import asyncio
import os
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from wildfire_msgs.msg import FireGrid
from cv_bridge import CvBridge


class ROSBridge(Node):

    def __init__(self, scout_agent, firefighter_agents):
        super().__init__('ros_bridge')

        self.scout_agent = scout_agent
        self.firefighter_agents = firefighter_agents
        self.bridge = CvBridge()

        # Bird's-eye image from viz_renderer (used by VLM)
        self.create_subscription(
            Image, '/viz/fire_image', self._camera_cb, 10)

        # Fire grid -> scout (non-VLM fallback path)
        self.create_subscription(
            FireGrid, '/fire_grid', self._fire_grid_cb, 10)

        # Per-firefighter subscriptions and publishers
        self.target_position_pubs = {}
        self.spray_enable_pubs = {}

        for ff_id, ff_agent in firefighter_agents.items():
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
            f'ROS Bridge ready — {len(firefighter_agents)} firefighters')

    # -- inbound callbacks ---------------------------------------------------

    def _camera_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.scout_agent.set_camera_image(cv_image)
        except Exception as e:
            self.get_logger().error(f'Camera conversion error: {e}')

    def _fire_grid_cb(self, msg: FireGrid):
        self.scout_agent.set_fire_grid(list(msg.intensity), msg.width)

    def _odom_cb(self, ff_id, msg):
        p = msg.pose.pose.position
        if ff_id in self.firefighter_agents:
            self.firefighter_agents[ff_id].update_position(p.x, p.y)

    def _water_cb(self, ff_id, msg):
        if ff_id in self.firefighter_agents:
            self.firefighter_agents[ff_id].update_water_level(msg.data)

    # -- outbound commands ---------------------------------------------------

    def _send_move(self, ff_id, position, fire_cell):
        pt = Point(x=float(position[0]), y=float(position[1]), z=0.0)
        if ff_id in self.target_position_pubs:
            self.target_position_pubs[ff_id].publish(pt)
            self.spray_enable_pubs[ff_id].publish(Bool(data=True))
            self.get_logger().info(
                f'Move {ff_id} -> ({position[0]:.1f}, {position[1]:.1f})')

    def _send_refill(self, ff_id, position):
        if ff_id in self.spray_enable_pubs:
            self.spray_enable_pubs[ff_id].publish(Bool(data=False))
        pt = Point(x=float(position[0]), y=float(position[1]), z=0.0)
        if ff_id in self.target_position_pubs:
            self.target_position_pubs[ff_id].publish(pt)
            self.get_logger().info(f'Refill {ff_id}')


def _run_agent(agent):
    """Run a uAgent in a fresh event loop on its own daemon thread."""
    import nest_asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)
    agent.run()


def main(args=None):
    from .scout_agent import ScoutAgent
    from .firefighter_agent import FirefighterAgent

    rclpy.init(args=args)

    num_firefighters = int(os.environ.get('NUM_FIREFIGHTERS', '4'))
    use_vlm = os.environ.get('USE_VLM', 'true').lower() == 'true'

    scout_agent = ScoutAgent(port=8000, use_vlm=use_vlm)

    firefighter_agents = {}
    for i in range(num_firefighters):
        ff_id = f'firefighter_{i + 1}'
        firefighter_agents[ff_id] = FirefighterAgent(
            firefighter_id=ff_id,
            port=8001 + i,
            scout_address=scout_agent.agent.address,
        )

    bridge = ROSBridge(scout_agent, firefighter_agents)

    # Start each uAgent in its own daemon thread
    threading.Thread(
        target=_run_agent, args=(scout_agent,), daemon=True).start()
    for ff in firefighter_agents.values():
        threading.Thread(
            target=_run_agent, args=(ff,), daemon=True).start()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
