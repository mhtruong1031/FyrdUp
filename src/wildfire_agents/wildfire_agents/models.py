#!/usr/bin/env python3
"""Shared uAgent message models used by firefighter agents."""

from typing import Tuple
from uagents import Model


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


class InFireAlert(Model):
    """Immediate ping: firefighter's grid cell is burning (needs rescue / replan)."""

    firefighter_id: str
    position: Tuple[float, float]
    water_level: float
    state: str
    reason: str  # e.g. on_burning_cell
