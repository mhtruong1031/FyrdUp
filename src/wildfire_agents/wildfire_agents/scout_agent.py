#!/usr/bin/env python3
"""
Scout ADK Agent — aerial coordinator for wildfire firefighting.

Uses Google ADK (Agent Development Kit) with Gemini to agentically:
  1. Render its conical field-of-view onto the fire grid
  2. Adjust position / altitude until all fire + firefighters are in view
  3. Allocate firefighters to optimal perimeter positions

The bridge calls ``run_analysis()`` on a timer; the ADK agent iterates
internally via tool calls until it is satisfied with coverage.
"""

from __future__ import annotations

import base64
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOV_HALF_ANGLE_DEG = 35.0
FOV_HALF_ANGLE_RAD = math.radians(FOV_HALF_ANGLE_DEG)
MIN_ALTITUDE = 5.0
MAX_ALTITUDE = 40.0
FOV_IMAGE_PX = 256  # rendered FOV image side length

SCOUT_INSTRUCTION = """\
You are a scout drone monitoring a wildfire from above.  You have a downward-
facing conical field of view that projects as a circle on the ground.

## Available tools

| Tool | Purpose |
|------|---------|
| get_current_fov | Capture what you see right now.  Returns structured data about visible fire cells, visible firefighters, coverage statistics, and a base64 PNG of the view. |
| move_scout | Set your absolute (x, y, z) position in world coordinates.  z is altitude. |
| adjust_altitude | Change altitude by a signed delta (positive = up). |
| get_fire_status | Global fire statistics: total burning cells, centroid, bounding box. |
| get_firefighter_positions | Positions, water levels, and states of every firefighter. |
| assign_firefighter | Command a specific firefighter to move to a world (x, y) position. |

## Goals (priority order)
1. Keep **ALL** active fire cells within your field of view.
2. Keep **ALL** firefighter robots within your field of view.
3. Minimise wasted (empty) space — fly lower when you can for better detail.

## Strategy
- Start by calling **get_current_fov()** to assess your current view.
- If any fire edges are clipped or firefighters are missing, either
  **move_scout** toward the centroid of the missing elements or
  **adjust_altitude** upward (+2 to +5 metres).
- If more than 60 % of the view is empty non-fire, non-firefighter space,
  **adjust_altitude** downward (−1 to −3) for higher resolution.
- After each adjustment call **get_current_fov()** again to verify.
- Once you are satisfied with coverage, call **get_fire_status** and
  **get_firefighter_positions**, then **assign_firefighter** for each
  available firefighter to an optimal fire-perimeter position.
- Distribute firefighters evenly around the perimeter; prefer upwind.
- When finished, reply with a short tactical summary.
"""


# ---------------------------------------------------------------------------
# Scout ADK Agent
# ---------------------------------------------------------------------------

class ScoutADKAgent:
    """Wraps a Google ADK Agent and the state needed by its tools."""

    def __init__(
        self,
        use_vlm: bool = True,
        analysis_interval: float = 10.0,
    ):
        self.use_vlm = use_vlm
        self.analysis_interval = analysis_interval

        # mutable state set by the ROS bridge
        self.fire_grid: Optional[List[float]] = None
        self.grid_size: int = 50
        self.cell_size: float = 1.0

        self.scout_position: List[float] = [0.0, 0.0, 15.0]  # x, y, z
        self.firefighter_status: Dict[str, dict] = {}

        # callbacks wired by the bridge
        self.on_move_scout: Optional[callable] = None       # (x, y, z)
        self.on_assign_firefighter: Optional[callable] = None  # (ff_id, x, y)

        self._build_adk_agent()

    # -- ADK agent construction ----------------------------------------------

    def _build_adk_agent(self):
        self_ref = self  # closure capture

        # ---- tool functions (plain functions with docstrings) ----

        def get_current_fov() -> dict:
            """Capture the scout's current conical field-of-view.

            Returns a dict with keys:
              scout_x, scout_y, scout_z,
              fov_radius_cells, fire_cells_in_fov, fire_cells_total,
              firefighters_in_fov (list of ids), firefighters_total,
              fire_coverage_pct, empty_pct,
              fire_centroid (x, y) in world coords or null,
              fire_clipped (bool — true if any fire is outside the FOV),
              fov_image_base64 (PNG)
            """
            return self_ref._compute_fov()

        def move_scout(x: float, y: float, z: float) -> dict:
            """Move the scout drone to absolute world position (x, y, z).

            Args:
                x: World x coordinate.
                y: World y coordinate.
                z: Altitude in metres (clamped to 5-40 m).
            """
            z = max(MIN_ALTITUDE, min(MAX_ALTITUDE, z))
            self_ref.scout_position = [x, y, z]
            if self_ref.on_move_scout:
                self_ref.on_move_scout(x, y, z)
            return {"status": "ok", "position": [x, y, z]}

        def adjust_altitude(delta_z: float) -> dict:
            """Raise (positive) or lower (negative) the scout by delta_z metres.

            Args:
                delta_z: Altitude change in metres.
            """
            new_z = max(MIN_ALTITUDE, min(MAX_ALTITUDE,
                                          self_ref.scout_position[2] + delta_z))
            self_ref.scout_position[2] = new_z
            if self_ref.on_move_scout:
                self_ref.on_move_scout(*self_ref.scout_position)
            return {"status": "ok", "altitude": new_z}

        def get_fire_status() -> dict:
            """Return global fire statistics: burning count, centroid, bounding box."""
            return self_ref._fire_status()

        def get_firefighter_positions() -> dict:
            """Return every firefighter's position, water level, and state."""
            out = {}
            for ff_id, st in self_ref.firefighter_status.items():
                out[ff_id] = {
                    "position": list(st["position"]),
                    "water_level": st["water_level"],
                    "state": st["state"],
                }
            return {"firefighters": out, "total": len(out)}

        def assign_firefighter(firefighter_id: str, target_x: float, target_y: float) -> dict:
            """Command a firefighter to move to a world (x, y) position.

            Args:
                firefighter_id: e.g. 'firefighter_1'.
                target_x: World x coordinate for the firefighter.
                target_y: World y coordinate for the firefighter.
            """
            if self_ref.on_assign_firefighter:
                self_ref.on_assign_firefighter(firefighter_id, target_x, target_y)
            return {"status": "ok", "firefighter_id": firefighter_id,
                    "target": [target_x, target_y]}

        model_name = os.environ.get("ADK_MODEL", "gemini-2.5-flash-lite")

        self.agent = Agent(
            model=model_name,
            name="scout_drone",
            description="Aerial scout drone that monitors a wildfire and coordinates ground firefighters.",
            instruction=SCOUT_INSTRUCTION,
            tools=[
                get_current_fov,
                move_scout,
                adjust_altitude,
                get_fire_status,
                get_firefighter_positions,
                assign_firefighter,
            ],
        )

        self._session_service = InMemorySessionService()
        self._runner = InMemoryRunner(self.agent, app_name="wildfire_scout")
        self._runner.session_service = self._session_service
        self._session_id = 0

    # -- public interface (called by ros_bridge timer) -----------------------

    async def run_analysis(self):
        """Run one agentic analysis cycle.  The ADK agent will call tools
        (get_current_fov, move_scout, adjust_altitude, assign_firefighter …)
        in a multi-step loop until it is satisfied with coverage.
        """
        if self.fire_grid is None:
            print("[SCOUT-ADK] No fire grid yet — skipping")
            return

        self._session_id += 1
        sid = f"analysis_{self._session_id}"

        await self._session_service.create_session(
            app_name="wildfire_scout",
            user_id="system",
            session_id=sid,
        )

        prompt = (
            "Analyse your current field of view.  Make sure all fire and "
            "all firefighters are visible.  Adjust your position or altitude "
            "as needed, then assign firefighters to optimal positions."
        )

        content = Content(parts=[Part(text=prompt)])

        final_text = ""
        try:
            async for event in self._runner.run_async(
                new_message=content,
                user_id="system",
                session_id=sid,
            ):
                self._log_event(event)
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = "".join(
                        p.text for p in event.content.parts if getattr(p, "text", None)
                    )
        except Exception as exc:
            print(f"[SCOUT-ADK] Agent run failed: {exc}")
            return

        if final_text:
            print(f"\n[SCOUT-ADK] === Final Summary ===\n{final_text}\n")

    def _log_event(self, event):
        """Print intermediate ADK events for visibility into scout reasoning."""
        author = getattr(event, "author", "?")

        # Tool calls issued by the agent
        calls = event.get_function_calls() if hasattr(event, "get_function_calls") else []
        for call in calls:
            args_pretty = json.dumps(
                call.args if isinstance(call.args, dict) else str(call.args),
                indent=2,
            )
            print(f"[SCOUT-ADK] Tool call: {call.name}(\n{args_pretty}\n)")

        # Tool responses
        responses = event.get_function_responses() if hasattr(event, "get_function_responses") else []
        for resp in responses:
            data = resp.response
            if isinstance(data, dict):
                filtered = {k: v for k, v in data.items() if k != "fov_image_base64"}
                pretty = json.dumps(filtered, indent=2)
            else:
                pretty = json.dumps(data, indent=2) if data is not None else str(data)
            print(f"[SCOUT-ADK] Tool result: {resp.name} →\n{pretty}\n")

        # Intermediate reasoning text from the model
        if (
            not event.is_final_response()
            and event.content
            and event.content.parts
            and author != "user"
        ):
            text = "".join(
                p.text for p in event.content.parts if getattr(p, "text", None)
            )
            if text.strip() and not calls:
                print(f"[SCOUT-ADK] Reasoning: {text.strip()}")

    # -- data setters (called by ros_bridge) ---------------------------------

    def set_fire_grid(self, intensity: list, grid_size: int):
        self.fire_grid = intensity
        self.grid_size = grid_size

    def set_camera_image(self, image: np.ndarray):
        pass  # ADK agent uses structured FOV, not raw camera images

    def update_scout_position(self, x: float, y: float, z: float):
        self.scout_position = [x, y, z]

    def update_firefighter(self, ff_id: str, position: Tuple[float, float],
                           water_level: float, state: str):
        self.firefighter_status[ff_id] = {
            "position": position,
            "water_level": water_level,
            "state": state,
        }

    # -- FOV computation -----------------------------------------------------

    def _compute_fov(self) -> dict:
        """Render the conical FOV and return structured + image data."""
        sx, sy, sz = self.scout_position
        n = self.grid_size
        cs = self.cell_size
        half = n * cs / 2.0

        fov_radius_world = sz * math.tan(FOV_HALF_ANGLE_RAD)
        fov_radius_cells = fov_radius_world / cs

        # fire stats
        fire_cells_total = 0
        fire_cells_in_fov = 0
        fire_xs, fire_ys = [], []
        all_fire_xs, all_fire_ys = [], []

        for gy in range(n):
            for gx in range(n):
                val = self.fire_grid[gy * n + gx]
                if val < 0.1:
                    continue
                fire_cells_total += 1
                wx = (gx - n / 2.0 + 0.5) * cs
                wy = (gy - n / 2.0 + 0.5) * cs
                all_fire_xs.append(wx)
                all_fire_ys.append(wy)
                dist = math.hypot(wx - sx, wy - sy)
                if dist <= fov_radius_world:
                    fire_cells_in_fov += 1
                    fire_xs.append(wx)
                    fire_ys.append(wy)

        fire_centroid = None
        if all_fire_xs:
            fire_centroid = [
                sum(all_fire_xs) / len(all_fire_xs),
                sum(all_fire_ys) / len(all_fire_ys),
            ]
        fire_clipped = fire_cells_in_fov < fire_cells_total

        # firefighter visibility
        ff_in_fov = []
        for ff_id, st in self.firefighter_status.items():
            fx, fy = st["position"]
            if math.hypot(fx - sx, fy - sy) <= fov_radius_world:
                ff_in_fov.append(ff_id)

        # coverage stats
        total_cells_in_fov = math.pi * fov_radius_cells ** 2
        occupied = fire_cells_in_fov + len(ff_in_fov)
        empty_pct = max(0.0, 1.0 - occupied / max(total_cells_in_fov, 1.0))
        fire_cov = (fire_cells_in_fov / fire_cells_total * 100.0) if fire_cells_total else 100.0

        # render FOV image
        img_b64 = self._render_fov_image(sx, sy, fov_radius_world)

        return {
            "scout_x": round(sx, 2),
            "scout_y": round(sy, 2),
            "scout_z": round(sz, 2),
            "fov_radius_cells": round(fov_radius_cells, 1),
            "fire_cells_in_fov": fire_cells_in_fov,
            "fire_cells_total": fire_cells_total,
            "fire_coverage_pct": round(fire_cov, 1),
            "fire_clipped": fire_clipped,
            "fire_centroid": fire_centroid,
            "firefighters_in_fov": ff_in_fov,
            "firefighters_total": len(self.firefighter_status),
            "empty_pct": round(empty_pct * 100, 1),
            "fov_image_base64": img_b64,
        }

    def _render_fov_image(self, cx: float, cy: float, radius: float) -> str:
        """Render a top-down grid image of the circular FOV region."""
        n = self.grid_size
        cs = self.cell_size
        px = FOV_IMAGE_PX
        half = n * cs / 2.0

        img = np.full((px, px, 3), (20, 50, 20), dtype=np.uint8)

        if radius < 0.01:
            return self._img_to_b64(img)

        for py_ in range(px):
            for px_ in range(px):
                # map pixel to world coords inside the FOV square
                wx = cx + (px_ / px - 0.5) * 2.0 * radius
                wy = cy + (py_ / px - 0.5) * 2.0 * radius
                if math.hypot(wx - cx, wy - cy) > radius:
                    img[py_, px_] = (10, 10, 10)
                    continue
                gx = int((wx + half) / cs)
                gy = int((wy + half) / cs)
                if not (0 <= gx < n and 0 <= gy < n):
                    img[py_, px_] = (10, 10, 10)
                    continue
                val = self.fire_grid[gy * n + gx]
                if val > 0.1:
                    r = min(255, int(val * 255))
                    g = max(0, int((1.0 - val) * 60))
                    img[py_, px_] = (r, g, 0)

        # draw firefighters
        for ff_id, st in self.firefighter_status.items():
            fx, fy = st["position"]
            if math.hypot(fx - cx, fy - cy) > radius:
                continue
            fpx = int(((fx - cx) / (2.0 * radius) + 0.5) * px)
            fpy = int(((fy - cy) / (2.0 * radius) + 0.5) * px)
            if 0 <= fpx < px and 0 <= fpy < px:
                cv2.circle(img, (fpx, fpy), 4, (0, 255, 255), -1)

        # crosshair at centre
        cv2.drawMarker(img, (px // 2, px // 2), (255, 255, 255),
                        cv2.MARKER_CROSS, 8, 1)

        return self._img_to_b64(img)

    @staticmethod
    def _img_to_b64(img: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf.tobytes()).decode()

    # -- fire status helper --------------------------------------------------

    def _fire_status(self) -> dict:
        n = self.grid_size
        cs = self.cell_size
        burning = 0
        xs, ys = [], []
        min_gx = n
        max_gx = 0
        min_gy = n
        max_gy = 0

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


# ---------------------------------------------------------------------------
# Standalone entry point (for testing outside the bridge)
# ---------------------------------------------------------------------------

def main():
    import asyncio as _aio
    use_vlm = os.environ.get("USE_VLM", "true").lower() == "true"
    scout = ScoutADKAgent(use_vlm=use_vlm)
    # Without the bridge we can't do much; just verify construction.
    print(f"[SCOUT-ADK] Agent created.  model={scout.agent.model}")


if __name__ == "__main__":
    main()
