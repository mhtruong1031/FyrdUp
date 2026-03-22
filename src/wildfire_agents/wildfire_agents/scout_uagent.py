#!/usr/bin/env python3
"""
Thin uAgent wrapper around the ADK scout agent.

Provides a real uAgent identity (port 8000) so firefighter uAgents can send
StatusUpdate and InFireAlert messages to the scout, and the bridge can dispatch
MoveCommand / RefillCommand messages to firefighters via the uAgent protocol.

By default the scout registers with an **Agentverse mailbox** URL (see
``SCOUT_UAGENT_MAILBOX``) so the agent is visible/discoverable there while the
local HTTP server still runs (uAgents keeps the server when REST/inspector
handlers exist).

All ADK intelligence lives in scout_agent.py — this module only handles
uAgent message plumbing.
"""

import queue

from uagents import Agent, Context

from typing import Callable, Optional

from .models import InFireAlert, MoveCommand, RefillCommand, StatusUpdate
from .uagent_env import env_flag


class ScoutUAgent:

    def __init__(self, adk_agent, port: int = 8000, seed: str = "scout_drone_seed"):
        self.adk_agent = adk_agent
        self.on_in_fire_alert: Optional[Callable] = None
        use_mailbox = env_flag('SCOUT_UAGENT_MAILBOX', True)
        self.agent = Agent(
            name="scout_drone_uagent",
            port=port,
            seed=seed,
            mailbox=use_mailbox,
        )
        self._ff_addresses: dict[str, str] = {}
        self._outbox: queue.Queue = queue.Queue()

        self._setup_handlers()

    @property
    def address(self) -> str:
        return self.agent.address

    def register_firefighter(self, ff_id: str, address: str):
        self._ff_addresses[ff_id] = address

    def send_move_command(
        self,
        ff_id: str,
        position: tuple[float, float],
        fire_cell: tuple[int, int],
        priority: int = 1,
    ):
        addr = self._ff_addresses.get(ff_id)
        if addr is None:
            return
        msg = MoveCommand(
            target_id=ff_id,
            position=position,
            priority=priority,
            fire_cell=fire_cell,
        )
        self._outbox.put((addr, msg))

    def send_refill_command(
        self,
        ff_id: str,
        refill_position: tuple[float, float] = (0.0, 0.0),
    ):
        addr = self._ff_addresses.get(ff_id)
        if addr is None:
            return
        msg = RefillCommand(
            target_id=ff_id,
            refill_position=refill_position,
        )
        self._outbox.put((addr, msg))

    def _setup_handlers(self):

        @self.agent.on_message(model=StatusUpdate)
        async def handle_status(ctx: Context, sender: str, msg: StatusUpdate):
            self.adk_agent.update_firefighter(
                msg.firefighter_id, msg.position, msg.water_level, msg.state,
            )

        @self.agent.on_message(model=InFireAlert)
        async def handle_in_fire(ctx: Context, sender: str, msg: InFireAlert):
            self.adk_agent.update_firefighter(
                msg.firefighter_id, msg.position, msg.water_level, msg.state,
            )
            cb = self.on_in_fire_alert
            if cb is not None:
                cb(msg)

        @self.agent.on_interval(period=0.5)
        async def drain_outbox(ctx: Context):
            while not self._outbox.empty():
                try:
                    addr, msg = self._outbox.get_nowait()
                    await ctx.send(addr, msg)
                except queue.Empty:
                    break

    def run(self):
        self.agent.run()
