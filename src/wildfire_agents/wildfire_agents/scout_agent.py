#!/usr/bin/env python3
"""
Scout Agent — aerial coordinator for wildfire firefighting.

Single-shot design: each analysis cycle computes all state locally,
positions the scout deterministically above the fire centroid, then
makes ONE Gemini API call to decide firefighter assignments.

The bridge calls ``run_analysis()`` on a timer.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import google.genai as genai
from google.genai import types

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOV_HALF_ANGLE_DEG = 35.0
FOV_HALF_ANGLE_RAD = math.radians(FOV_HALF_ANGLE_DEG)
MIN_ALTITUDE = 5.0
MAX_ALTITUDE = 40.0

REFILL_WATER_THRESHOLD = 20.0
FIRE_CELL_THRESHOLD = 0.1

ASSIGNMENT_PROMPT = """\
You are a wildfire tactical coordinator.  Given the current fire state and
firefighter positions, decide what each firefighter should do.

## Rules
- NEVER send a firefighter onto a burning cell.
- Only assign firefighters to positions from the **perimeter** list below.
- If a firefighter's water_level < 20, it MUST refill (action="refill").
- If a firefighter is MOVING, leave it alone (action="hold") unless its
  target area is no longer on fire.
- If a firefighter is FIGHTING and there is still fire near it, keep it
  there (action="hold").
- If a firefighter is IDLE or its fire is out, assign it to an
  unassigned perimeter position (action="move").

## Spacing — default behavior (when `rescue.active` is false)
- **Under normal circumstances, spreading out is mandatory.** Your default
  goal is to place firefighters around the fire **perimeter** so they cover
  different sectors of the blaze, not stacked on the same side.
- Maximize **angular separation** around the fire centroid: imagine the
  perimeter as a ring—pick targets that are far apart along that ring.
- Maximize **pairwise distance** between firefighters: avoid clustering;
  if two units are already close, assign moves to **opposite** or **distant**
  perimeter arcs when you issue new positions.
- Prefer perimeter points in **empty** wedges (no teammate nearby along
  the fire edge) before reusing a congested sector.
- Do not assign two robots to the same perimeter coordinate; avoid adjacent
  perimeter cells when other valid coordinates exist.

## Rescue (highest priority when `rescue.active` is true)
- Some units may be **in_danger** (on a burning cell or encircled by fire).
  They often cannot path out because every route is blocked by fire.
- When `rescue.active` is true, you MUST task **at least one** other
  firefighter who has water_level >= 20 and is not already critical with
  a **move** to a perimeter point from the list that is **closest** to the
  trapped teammate's position (among perimeter coordinates), so they can
  spray and shrink the fire next to that teammate. Use two rescuers if
  multiple teammates are in danger or one rescuer is far away.
- Prefer rescuers who are IDLE or FIGHTING in a low-threat sector; you may
  override a simple "hold" for FIGHTING rescuers if they are the best
  option to save someone in_danger.
- Still never assign any move target onto a burning cell; only use
  perimeter list coordinates.
- Trapped units: if they have water and are in_danger, usually **hold** so
  they keep spraying; if water_level < 20, **refill** only if a safe path
  exists conceptually—when impossible, prioritize others clearing fire
  toward them first (move rescuers closer).
- After rescue needs are addressed, return to **wide spacing** for everyone
  you move (same rules as the Spacing section).

## State
{state_json}

## Response format — ONLY valid JSON, no markdown fences
{{
  "assignments": [
    {{"firefighter_id": "firefighter_1", "action": "move", "target": [x, y]}},
    {{"firefighter_id": "firefighter_2", "action": "refill"}},
    {{"firefighter_id": "firefighter_3", "action": "hold"}}
  ],
  "scout_summary": "one sentence tactical note"
}}

"action" must be one of: "move", "refill", "hold".
"target" is required only when action is "move" and MUST be a coordinate
pair from the perimeter list.
"""


class ScoutADKAgent:
    """Single-shot scout agent: deterministic positioning + one LLM call."""

    def __init__(
        self,
        use_vlm: bool = True,
        analysis_interval: float = 20.0,
    ):
        self.use_vlm = use_vlm
        self.analysis_interval = analysis_interval

        self.fire_grid: Optional[List[float]] = None
        self.grid_size: int = 50
        self.cell_size: float = 1.0

        self.scout_position: List[float] = [0.0, 0.0, 15.0]
        self.firefighter_status: Dict[str, dict] = {}

        self.water_supply_position = [0.0, 0.0]

        # Callbacks wired by the bridge
        self.on_move_scout: Optional[callable] = None
        self.on_assign_firefighter: Optional[callable] = None
        self.on_refill_firefighter: Optional[callable] = None

        self._model_name = os.environ.get("ADK_MODEL", "gemini-2.5-flash-lite")
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)

    # -- public interface (called by ros_bridge timer) -----------------------

    async def run_analysis(self):
        """One analysis cycle: reposition scout, call Gemini once, dispatch."""
        if self.fire_grid is None:
            print("[SCOUT] No fire grid yet — skipping")
            return

        fire_status = self._fire_status()
        if fire_status["burning_cells"] == 0:
            print("[SCOUT] No active fire — skipping")
            return

        self._auto_position_scout(fire_status)

        perimeter = self._fire_perimeter()
        ff_positions = self._get_firefighter_positions()

        # Fast-path: handle obvious refills and idle assignments without LLM
        # if all decisions are deterministic
        assignments = self._try_deterministic_assignments(
            fire_status, perimeter, ff_positions)
        summary = None

        if assignments is None:
            assignments, summary = await self._call_gemini(
                fire_status, perimeter, ff_positions)

        self._execute_assignments(assignments, perimeter)

        if summary:
            print(f"[SCOUT] {summary}")

    # -- deterministic scout positioning ------------------------------------

    def _auto_position_scout(self, fire_status: dict):
        """Move scout above fire centroid at altitude that covers all fire."""
        centroid = fire_status.get("centroid")
        bbox = fire_status.get("bounding_box")
        if centroid is None or bbox is None:
            return

        cx, cy = centroid

        fire_span = max(
            bbox["max_x"] - bbox["min_x"],
            bbox["max_y"] - bbox["min_y"],
        )
        margin = 5.0
        required_radius = fire_span / 2.0 + margin

        # Also ensure all firefighters are in FOV
        for st in self.firefighter_status.values():
            fx, fy = st["position"]
            dist = math.hypot(fx - cx, fy - cy)
            required_radius = max(required_radius, dist + 2.0)

        required_alt = required_radius / math.tan(FOV_HALF_ANGLE_RAD)
        z = max(MIN_ALTITUDE, min(MAX_ALTITUDE, required_alt))

        self.scout_position = [cx, cy, z]
        if self.on_move_scout:
            self.on_move_scout(cx, cy, z)

    # -- deterministic fast-path --------------------------------------------

    def _try_deterministic_assignments(
        self, fire_status, perimeter, ff_positions
    ) -> Optional[list]:
        """Return assignments list if all decisions are obvious, else None."""
        if not self.firefighter_status:
            return None

        for st in self.firefighter_status.values():
            if self._firefighter_in_danger(st["position"])[0]:
                return None

        perim_pts = perimeter.get("perimeter", [])
        if not perim_pts and fire_status["burning_cells"] > 0:
            return None

        assignments = []
        used_perimeter = set()

        for ff_id, st in self.firefighter_status.items():
            wl = st["water_level"]
            state = st["state"]

            if wl < REFILL_WATER_THRESHOLD:
                assignments.append({
                    "firefighter_id": ff_id, "action": "refill"})
                continue

            if state == "MOVING":
                assignments.append({
                    "firefighter_id": ff_id, "action": "hold"})
                continue

            if state == "FIGHTING":
                fx, fy = st["position"]
                still_near_fire = self._near_fire(fx, fy)
                if still_near_fire:
                    assignments.append({
                        "firefighter_id": ff_id, "action": "hold"})
                    continue
                # Fire cleared — fall through to reassign

            if state in ("IDLE", "FIGHTING", "REFILLING") and perim_pts:
                best_idx = self._spread_pick_perimeter(
                    ff_id, st["position"], perim_pts, used_perimeter)
                if best_idx is not None:
                    used_perimeter.add(best_idx)
                    assignments.append({
                        "firefighter_id": ff_id,
                        "action": "move",
                        "target": perim_pts[best_idx],
                    })
                    continue

            # Can't decide deterministically
            return None

        return assignments

    def _near_fire(self, x: float, y: float, radius: float = 3.0) -> bool:
        n = self.grid_size
        cs = self.cell_size
        half = n * cs / 2.0
        gx = int((x + half) / cs)
        gy = int((y + half) / cs)
        r_cells = int(radius / cs) + 1
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < n and 0 <= ny < n:
                    if self.fire_grid[ny * n + nx] > 0.1:
                        return True
        return False

    def _spread_pick_perimeter(
        self,
        ff_id: str,
        pos: tuple,
        perimeter: list,
        used: set,
    ) -> Optional[int]:
        """Pick a perimeter index that spreads agents around the fire (maximin from peers)."""
        px, py = pos
        others = [
            st["position"]
            for oid, st in self.firefighter_status.items()
            if oid != ff_id
        ]
        best_i = None
        best_key = None
        for i, (wx, wy) in enumerate(perimeter):
            if i in used:
                continue
            d_self = math.hypot(wx - px, wy - py)
            if others:
                d_sep = min(
                    math.hypot(wx - ox, wy - oy) for ox, oy in others)
            else:
                d_sep = float("inf")
            # Prefer large separation; break ties with shorter travel.
            key = (d_sep, -d_self)
            if best_key is None or key > best_key:
                best_key = key
                best_i = i
        return best_i

    # -- single Gemini API call ---------------------------------------------

    async def _call_gemini(self, fire_status, perimeter, ff_positions):
        """One LLM call: state in → assignments out."""
        ffs = ff_positions["firefighters"]
        in_trouble = [
            {
                "firefighter_id": fid,
                "position": data["position"],
                "water_level": data["water_level"],
                "state": data["state"],
                "danger_reason": data.get("danger_reason"),
            }
            for fid, data in ffs.items()
            if data.get("in_danger")
        ]
        state = {
            "fire": fire_status,
            "perimeter_positions": perimeter["perimeter"][:60],
            "perimeter_count": perimeter["count"],
            "firefighters": ffs,
            "rescue": {
                "active": len(in_trouble) > 0,
                "units_in_danger": in_trouble,
            },
        }

        prompt = ASSIGNMENT_PROMPT.format(state_json=json.dumps(state, indent=2))

        print(f"[SCOUT] Calling Gemini ({self._model_name}) …")
        try:
            import asyncio
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=900,
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = response.text.strip()
            print(f"[SCOUT] Gemini response ({len(text)} chars)")

            result = json.loads(text)
            assignments = result.get("assignments", [])
            summary = result.get("scout_summary", "")
            return assignments, summary

        except Exception as exc:
            print(f"[SCOUT] Gemini call failed: {exc}")
            return [], None

    # -- execute assignments ------------------------------------------------

    def _execute_assignments(self, assignments: list, perimeter: dict):
        perim_pts = perimeter.get("perimeter", [])

        for cmd in assignments:
            ff_id = cmd.get("firefighter_id", "")
            action = cmd.get("action", "hold")

            if action == "hold":
                continue

            if action == "refill":
                print(f"[SCOUT] {ff_id} → refill")
                if self.on_refill_firefighter:
                    self.on_refill_firefighter(ff_id)
                continue

            if action == "move":
                target = cmd.get("target")
                if not target or len(target) < 2:
                    continue
                tx, ty = float(target[0]), float(target[1])
                tx, ty, snapped = self._snap_off_fire(tx, ty)
                if snapped:
                    print(f"[SCOUT] {ff_id} target snapped off fire → ({tx:.1f}, {ty:.1f})")
                print(f"[SCOUT] {ff_id} → move ({tx:.1f}, {ty:.1f})")
                if self.on_assign_firefighter:
                    self.on_assign_firefighter(ff_id, tx, ty)

    # -- data setters (called by ros_bridge) ---------------------------------

    def set_fire_grid(self, intensity: list, grid_size: int):
        self.fire_grid = intensity
        self.grid_size = grid_size

    def set_camera_image(self, image: np.ndarray):
        pass

    def update_scout_position(self, x: float, y: float, z: float):
        self.scout_position = [x, y, z]

    def update_firefighter(self, ff_id: str, position: Tuple[float, float],
                           water_level: float, state: str):
        self.firefighter_status[ff_id] = {
            "position": position,
            "water_level": water_level,
            "state": state,
        }

    # -- helpers (fire status, perimeter, snap) ------------------------------

    def _grid_indices(self, x: float, y: float) -> tuple[int, int]:
        n = self.grid_size
        cs = self.cell_size
        half = n * cs / 2.0
        gx = int((x + half) / cs)
        gy = int((y + half) / cs)
        return gx, gy

    def _cell_burning(self, gx: int, gy: int) -> bool:
        if self.fire_grid is None:
            return False
        n = self.grid_size
        if not (0 <= gx < n and 0 <= gy < n):
            return False
        return self.fire_grid[gy * n + gx] >= FIRE_CELL_THRESHOLD

    def _firefighter_in_danger(self, pos: tuple) -> tuple[bool, Optional[str]]:
        """Stuck-in-fire heuristic: on fire, or tight ring of fire around cell."""
        x, y = pos
        gx, gy = self._grid_indices(x, y)
        if self._cell_burning(gx, gy):
            return True, "on_burning_cell"
        burning_nb = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if self._cell_burning(gx + dx, gy + dy):
                    burning_nb += 1
        if burning_nb >= 5:
            return True, "surrounded_by_fire"
        return False, None

    def _get_firefighter_positions(self) -> dict:
        out = {}
        for ff_id, st in self.firefighter_status.items():
            x, y = st["position"]
            in_danger, reason = self._firefighter_in_danger((x, y))
            out[ff_id] = {
                "position": list(st["position"]),
                "water_level": st["water_level"],
                "state": st["state"],
                "in_danger": in_danger,
                "danger_reason": reason,
            }
        return {"firefighters": out, "total": len(out)}

    def _fire_status(self) -> dict:
        n = self.grid_size
        cs = self.cell_size
        burning = 0
        xs, ys = [], []
        min_gx, max_gx = n, 0
        min_gy, max_gy = n, 0

        for gy in range(n):
            for gx in range(n):
                val = self.fire_grid[gy * n + gx]
                if val < 0.1:
                    continue
                burning += 1
                wx = (gx - n / 2.0 + 0.5) * cs
                wy = (gy - n / 2.0 + 0.5) * cs
                xs.append(wx)
                ys.append(wy)
                min_gx = min(min_gx, gx)
                max_gx = max(max_gx, gx)
                min_gy = min(min_gy, gy)
                max_gy = max(max_gy, gy)

        centroid = None
        bbox = None
        if xs:
            centroid = [round(sum(xs) / len(xs), 2),
                        round(sum(ys) / len(ys), 2)]
            bbox = {
                "min_x": round((min_gx - n / 2.0) * cs, 2),
                "max_x": round((max_gx - n / 2.0 + 1) * cs, 2),
                "min_y": round((min_gy - n / 2.0) * cs, 2),
                "max_y": round((max_gy - n / 2.0 + 1) * cs, 2),
            }

        threat = "low" if burning < 10 else ("medium" if burning < 30 else "high")
        return {
            "burning_cells": burning,
            "total_cells": n * n,
            "centroid": centroid,
            "bounding_box": bbox,
            "threat_level": threat,
        }

    def _fire_perimeter(self) -> dict:
        if self.fire_grid is None:
            return {"perimeter": [], "count": 0}

        n = self.grid_size
        cs = self.cell_size
        perimeter_set: set[tuple[int, int]] = set()

        for gy in range(n):
            for gx in range(n):
                if self.fire_grid[gy * n + gx] < 0.1:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx_, ny_ = gx + dx, gy + dy
                    if 0 <= nx_ < n and 0 <= ny_ < n:
                        if self.fire_grid[ny_ * n + nx_] < 0.1:
                            perimeter_set.add((nx_, ny_))

        perimeter_world = []
        for gx, gy in perimeter_set:
            wx = round((gx - n / 2.0 + 0.5) * cs, 2)
            wy = round((gy - n / 2.0 + 0.5) * cs, 2)
            perimeter_world.append([wx, wy])

        return {"perimeter": perimeter_world, "count": len(perimeter_world)}

    def _snap_off_fire(self, x: float, y: float) -> tuple[float, float, bool]:
        if self.fire_grid is None:
            return x, y, False

        n = self.grid_size
        cs = self.cell_size
        half = n * cs / 2.0
        gx = int((x + half) / cs)
        gy = int((y + half) / cs)

        if not (0 <= gx < n and 0 <= gy < n):
            return x, y, False
        if self.fire_grid[gy * n + gx] < 0.1:
            return x, y, False

        perim = self._fire_perimeter()["perimeter"]
        if not perim:
            return x, y, False

        best = min(perim, key=lambda p: math.hypot(p[0] - x, p[1] - y))
        return best[0], best[1], True


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    scout = ScoutADKAgent()
    print(f"[SCOUT] Agent created. model={scout._model_name}")


if __name__ == "__main__":
    main()
