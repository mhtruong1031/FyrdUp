#!/usr/bin/env python3
"""
Firefighter uAgent — ground robot that receives commands from the scout.

State machine: IDLE -> MOVING -> FIGHTING -> REFILLING -> IDLE
"""

import sys
import time
from uagents import Agent, Context

from .scout_agent import MoveCommand, RefillCommand, StatusUpdate


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
        self._last_pos_log_time = 0.0

        # Callbacks wired by ros_bridge
        self.on_move_command = None
        self.on_refill_command = None

        self._setup_handlers()

    def _setup_handlers(self):
        tag = f'[FF:{self.firefighter_id}]'

        @self.agent.on_interval(period=5.0)
        async def send_status(ctx: Context):
            status = StatusUpdate(
                firefighter_id=self.firefighter_id,
                position=self.position,
                water_level=self.water_level,
                state=self.state,
            )
            await ctx.send(self.scout_address, status)
            ctx.logger.info(
                f'{tag} StatusUpdate sent to scout '
                f'addr={self.scout_address[:20]}... | '
                f'state={self.state} water={self.water_level:.0f}% '
                f'pos=({self.position[0]:.1f}, {self.position[1]:.1f}) '
                f'target={self.target_position} '
                f'callbacks_wired={self.on_move_command is not None}')

        @self.agent.on_message(model=MoveCommand)
        async def handle_move(ctx: Context, sender: str, msg: MoveCommand):
            ctx.logger.info(
                f'{tag} MoveCommand RECEIVED from {sender[:20]}... '
                f'target_id={msg.target_id} '
                f'pos=({msg.position[0]:.1f}, {msg.position[1]:.1f}) '
                f'priority={msg.priority} fire_cell={msg.fire_cell}')
            if msg.target_id != self.firefighter_id:
                ctx.logger.warning(
                    f'{tag} IGNORING MoveCommand — target_id={msg.target_id} '
                    f'!= my_id={self.firefighter_id}')
                return
            old_state = self.state
            self.target_position = msg.position
            self.state = 'MOVING'
            ctx.logger.info(
                f'{tag} State {old_state} -> MOVING | '
                f'target=({msg.position[0]:.1f}, {msg.position[1]:.1f})')
            if self.on_move_command:
                ctx.logger.info(f'{tag} Invoking on_move_command callback (ros_bridge)')
                self.on_move_command(msg.position, msg.fire_cell)
            else:
                ctx.logger.warning(
                    f'{tag} on_move_command callback is NOT wired! '
                    f'ROS target_position will NOT be published.')

        @self.agent.on_message(model=RefillCommand)
        async def handle_refill(ctx: Context, sender: str, msg: RefillCommand):
            ctx.logger.info(
                f'{tag} RefillCommand RECEIVED from {sender[:20]}... '
                f'target_id={msg.target_id} '
                f'refill_pos=({msg.refill_position[0]:.1f}, '
                f'{msg.refill_position[1]:.1f})')
            if msg.target_id != self.firefighter_id:
                ctx.logger.warning(
                    f'{tag} IGNORING RefillCommand — target_id={msg.target_id} '
                    f'!= my_id={self.firefighter_id}')
                return
            old_state = self.state
            self.state = 'REFILLING'
            self.target_position = msg.refill_position
            ctx.logger.info(
                f'{tag} State {old_state} -> REFILLING | '
                f'target=({msg.refill_position[0]:.1f}, '
                f'{msg.refill_position[1]:.1f})')
            if self.on_refill_command:
                ctx.logger.info(f'{tag} Invoking on_refill_command callback (ros_bridge)')
                self.on_refill_command(msg.refill_position)
            else:
                ctx.logger.warning(
                    f'{tag} on_refill_command callback is NOT wired!')

    # -- data setters (called by ros_bridge) ---------------------------------

    def update_position(self, x: float, y: float):
        self.position = (x, y)
        if self.target_position is not None:
            dist = (
                (self.position[0] - self.target_position[0]) ** 2 +
                (self.position[1] - self.target_position[1]) ** 2
            ) ** 0.5

            now = time.monotonic()
            if now - self._last_pos_log_time >= 2.0:
                self._last_pos_log_time = now
                print(
                    f'[FF:{self.firefighter_id}] pos=({x:.1f}, {y:.1f}) '
                    f'target=({self.target_position[0]:.1f}, '
                    f'{self.target_position[1]:.1f}) '
                    f'dist={dist:.2f} state={self.state}')

            if dist < 0.5:
                old_state = self.state
                if self.state == 'MOVING':
                    self.state = 'FIGHTING'
                elif self.state == 'REFILLING':
                    self.state = 'IDLE'
                print(
                    f'[FF:{self.firefighter_id}] ARRIVED — '
                    f'state {old_state} -> {self.state} (dist={dist:.2f})')
                self.target_position = None

    def update_water_level(self, level: float):
        self.water_level = level

    def update_state(self, state: str):
        self.state = state

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
