#!/usr/bin/env python3
"""
World initialization from wildfire.png.

- Depth Anything V2 Small → 50×50 terrain heightmap
- Gemini 2.0 Flash VLM   → 12×12 binary fire shape
"""

import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Depth Anything heightmap
# ---------------------------------------------------------------------------

_heightmap_cache: dict = {}


def compute_heightmap(image_path: str, grid_size: int = 50) -> np.ndarray:
    """Run Depth Anything V2 Small on *image_path* and return a (grid_size, grid_size) heightmap.

    Each tile's height is the average predicted depth inside that tile.
    The result is normalised to [0, TERRAIN_HEIGHT_AMP] metres.
    """
    cache_key = (image_path, grid_size)
    if cache_key in _heightmap_cache:
        return _heightmap_cache[cache_key]

    TERRAIN_HEIGHT_AMP = 2.0

    from transformers import pipeline as hf_pipeline

    img = Image.open(image_path).convert("RGB")

    pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    result = pipe(img)
    depth_pil: Image.Image = result["depth"]

    depth_arr = np.array(depth_pil, dtype=np.float64)

    h, w = depth_arr.shape[:2]
    tile_h = h / grid_size
    tile_w = w / grid_size

    heightmap = np.zeros((grid_size, grid_size), dtype=np.float64)
    for gy in range(grid_size):
        y0 = int(gy * tile_h)
        y1 = int((gy + 1) * tile_h)
        for gx in range(grid_size):
            x0 = int(gx * tile_w)
            x1 = int((gx + 1) * tile_w)
            heightmap[gy, gx] = depth_arr[y0:y1, x0:x1].mean()

    lo, hi = heightmap.min(), heightmap.max()
    if hi - lo > 1e-6:
        heightmap = (heightmap - lo) / (hi - lo)
    else:
        heightmap[:] = 0.0
    heightmap *= TERRAIN_HEIGHT_AMP

    _heightmap_cache[cache_key] = heightmap
    print(f"[WORLD_INIT] Heightmap ready: shape={heightmap.shape} "
          f"range=[{heightmap.min():.2f}, {heightmap.max():.2f}]")
    return heightmap


# ---------------------------------------------------------------------------
# Gemini fire-shape detection
# ---------------------------------------------------------------------------

_fire_cache: dict = {}

FIRE_SHAPE_PROMPT = """\
You are analysing an aerial photograph of a wildfire.

Look at the image and identify where fire (flames, glowing embers, bright
orange/red regions) is located.

Represent the fire's shape on a **12×12 binary grid** where:
  1 = fire present in that cell
  0 = no fire

Row 0 is the TOP of the image, row 11 is the BOTTOM.
Column 0 is the LEFT, column 11 is the RIGHT.

Respond ONLY with valid JSON matching this schema (no markdown fences):
{
  "grid": [
    [0,0,0,0,0,0,0,0,0,0,0,0],
    ...
  ]
}

The "grid" array must have exactly 12 rows, each with exactly 12 integers (0 or 1).
"""


def detect_fire_shape(image_path: str) -> List[Tuple[int, int]]:
    """Use Gemini to detect the fire shape in *image_path*.

    Returns a list of (local_x, local_y) positions within a 12x12 block
    where fire is present.
    """
    if image_path in _fire_cache:
        return _fire_cache[image_path]

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)

    img = Image.open(image_path)
    img_bytes = _pil_to_jpeg_bytes(img)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=FIRE_SHAPE_PROMPT),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=800,
            temperature=0.1,
        ),
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    data = json.loads(text)
    grid = data["grid"]

    fire_positions: List[Tuple[int, int]] = []
    for row_idx, row in enumerate(grid):
        for col_idx, val in enumerate(row):
            if val:
                fire_positions.append((col_idx, row_idx))

    _fire_cache[image_path] = fire_positions
    print(f"[WORLD_INIT] Fire shape detected: {len(fire_positions)} cells "
          f"in 12x12 grid")
    return fire_positions


def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Convenience: resolve default image path
# ---------------------------------------------------------------------------

def default_image_path() -> str:
    """Return the path to wildfire.png, checking env var and common locations."""
    pkg_dir = os.path.dirname(__file__)
    candidates = [
        os.environ.get("WILDFIRE_IMAGE_PATH", ""),
        # source tree: src/wildfire_agents/wildfire_agents/ → 4 levels up
        os.path.join(pkg_dir, "..", "..", "..", "..", "wildfire.png"),
        # install tree: install/wildfire_agents/lib/python*/site-packages/wildfire_agents/
        os.path.join(pkg_dir, "..", "..", "..", "..", "..", "..", "wildfire.png"),
        # cwd fallback
        os.path.join(os.getcwd(), "wildfire.png"),
        # hardcoded known workspace path
        "/Users/minhtruong/Documents/BeachHacks2026/wildfire.png",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "wildfire.png not found. Set WILDFIRE_IMAGE_PATH or place it in the workspace root."
    )
