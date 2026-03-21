#!/usr/bin/env python3
"""
Gemini 2.0 Flash VLM for aerial fire analysis.

Replaces the GPT-4o implementation with Google's Gemini model.
Expects GOOGLE_API_KEY environment variable to be set.
"""

import asyncio
import base64
import json
import os

import cv2
import numpy as np
from typing import Dict

import google.generativeai as genai

from .vlm_interface import VLMInterface

TACTICAL_PROMPT = """\
You are the tactical AI for a scout drone in a wildfire firefighting simulation.
Analyze this aerial (bird's-eye) view and provide tactical recommendations.

The area is a 20x20 meter grid with 1-meter cells (coordinates 0-19 in x and y).
The water supply is located at the grid center (coordinates 10, 10 in grid space).

Red/orange cells represent fire. Green cells are unburned ground.
Yellow circles are firefighters. Blue square is the water supply.

Identify:
1. **Fire locations**: Grid (x, y) coordinates where fire is burning.
2. **Fire intensity**: For each location, estimate intensity on 0.0-1.0 scale.
3. **Wind direction**: Estimate from fire spread shape (0°=N, 90°=E, 180°=S, 270°=W).
4. **Recommended positions**: Optimal positions for firefighters to attack the fire.
   - Position firefighters on the perimeter of the fire, upwind if possible.
   - Distribute to maximize coverage; avoid clustering.
5. **Threat level**: Overall assessment ('low', 'medium', 'high').
6. **Analysis**: Brief tactical summary (2-3 sentences).

Respond ONLY with valid JSON matching this schema:
{
    "fire_locations": [[x1, y1], [x2, y2], ...],
    "fire_intensity": [0.8, 0.9, ...],
    "wind_direction": 45.0,
    "recommended_positions": [[x1, y1], [x2, y2], ...],
    "threat_level": "medium",
    "analysis": "Fire spreading northeast ..."
}

IMPORTANT: Coordinates must be integers 0-19.
fire_intensity array length must match fire_locations length.
"""


class GeminiVLM(VLMInterface):
    """Gemini 2.0 Flash implementation of the VLM interface."""

    def __init__(self, api_key: str = None, model_name: str = 'gemini-2.0-flash'):
        if api_key is None:
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError(
                    'Google API key not provided. '
                    'Set GOOGLE_API_KEY environment variable or pass api_key.'
                )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    async def analyze_fire(self, image: np.ndarray) -> Dict:
        try:
            jpeg_bytes = self._encode_image(image)

            image_part = {
                'mime_type': 'image/jpeg',
                'data': jpeg_bytes,
            }

            response = await asyncio.to_thread(
                self.model.generate_content,
                [TACTICAL_PROMPT, image_part],
                generation_config=genai.GenerationConfig(
                    response_mime_type='application/json',
                    max_output_tokens=1500,
                    temperature=0.2,
                ),
            )

            text = response.text.strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)

            if self._validate_response(result):
                return result

            print('Warning: Invalid VLM response, using defaults')
            return self._create_default_response()

        except Exception as e:
            print(f'Error in Gemini VLM analysis: {e}')
            return self._create_default_response()

    def _encode_image(self, image: np.ndarray) -> bytes:
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        success, buffer = cv2.imencode('.jpg', image_bgr)
        if not success:
            raise ValueError('Failed to encode image as JPEG')

        return buffer.tobytes()


async def test_gemini_vlm():
    """Quick smoke test with a dummy image."""
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    vlm = GeminiVLM()
    result = await vlm.analyze_fire(test_image)
    print('VLM Analysis Result:')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    asyncio.run(test_gemini_vlm())
