#!/usr/bin/env python3
"""
Firefighter uAgent — ground robot that receives commands from the scout.

State machine: IDLE -> MOVING -> FIGHTING -> REFILLING -> IDLE
"""

import queue
import sys
from uagents import Agent, Context

from .models import InFireAlert, MoveCommand, RefillCommand, StatusUpdate


class FirefighterAgent:

    def __init__(self, firefighter_id: str, port: int, scout_address: str):
        self.firefighter_id = firefighter_id
        self.agent = Agent(
            name=firefighter_id,
            port=port,
            seed=f'{firefighter_id}_seed',
        )
        self.scout_address = scout_address

        self.position = (0.0, -12.0)
        self.water_level = 100.0
        self.state = 'IDLE'
        self.target_position = None
        self.fire_cell = None

        # Callbacks wired by ros_bridge
        self.on_move_command = None
        self.on_refill_command = None

        self._in_fire_outbox: queue.Queue = queue.Queue(maxsize=4)

        self._setup_handlers()

    def _setup_handlers(self):

        @self.agent.on_interval(period=5.0)
        async def send_status(ctx: Context):
            status = StatusUpdate(
                firefighter_id=self.firefighter_id,
                position=self.position,
                water_level=self.water_level,
                state=self.state,
            )
            try:
                await ctx.send(self.scout_address, status)
            except Exception:
                pass  # scout may not have a uAgent endpoint

        @self.agent.on_interval(period=0.25)
        async def drain_in_fire_alerts(ctx: Context):
            while True:
                try:
                    reason = self._in_fire_outbox.get_nowait()
                except queue.Empty:
                    break
                alert = InFireAlert(
                    firefighter_id=self.firefighter_id,
                    position=self.position,
                    water_level=self.water_level,
                    state=self.state,
                    reason=reason,
                )
                try:
                    await ctx.send(self.scout_address, alert)
                except Exception:
                    pass

        @self.agent.on_message(model=MoveCommand)
        async def handle_move(ctx: Context, sender: str, msg: MoveCommand):
            if msg.target_id != self.firefighter_id:
                return
            self.target_position = msg.position
            self.fire_cell = msg.fire_cell
            self.state = 'MOVING'
            if self.on_move_command:
                self.on_move_command(msg.position, msg.fire_cell)

        @self.agent.on_message(model=RefillCommand)
        async def handle_refill(ctx: Context, sender: str, msg: RefillCommand):
            if msg.target_id != self.firefighter_id:
                return
            self.state = 'REFILLING'
            self.target_position = msg.refill_position
            if self.on_refill_command:
                self.on_refill_command(msg.refill_position)

    # -- data setters (called by ros_bridge) ---------------------------------

    def update_position(self, x: float, y: float):
        self.position = (x, y)
        if self.target_position is not None:
            dist = (
                (self.position[0] - self.target_position[0]) ** 2 +
                (self.position[1] - self.target_position[1]) ** 2
            ) ** 0.5

            if dist < 0.5:
                if self.state == 'MOVING':
                    self.state = 'FIGHTING'
                elif self.state == 'REFILLING':
                    self.state = 'IDLE'
                self.target_position = None

    def update_water_level(self, level: float):
        self.water_level = level

    def update_state(self, state: str):
        self.state = state

    def enqueue_in_fire_alert(self, reason: str = "on_burning_cell"):
        """Thread-safe: ROS bridge enqueues; uAgent interval sends to scout."""
        try:
            self._in_fire_outbox.put_nowait(reason)
        except queue.Full:
            pass

    def run(self):
        self.agent.run()


def main():
    if len(sys.argv) < 3:
        print('Usage: firefighter_agent.py <firefighter_id> <port>')
        sys.exit(1)

    firefighter_id = sys.argv[1]
    port = int(sys.argv[2])
    scout_address = 'placeholder'

    agent = FirefighterAgent(firefighter_id, port, scout_address)
    agent.run()


if __name__ == '__main__':
    main()
