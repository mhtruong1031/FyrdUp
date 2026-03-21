#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class VLMInterface(ABC):
    """Base class for Vision Language Model interfaces."""

    @abstractmethod
    async def analyze_fire(self, image: np.ndarray) -> Dict:
        """
        Analyze fire from aerial image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Dictionary containing:
            {
                'fire_locations': [[x, y], ...],       # Grid coordinates
                'fire_intensity': [0.0-1.0, ...],      # Per location
                'wind_direction': float,                # Degrees (0=N, 90=E)
                'recommended_positions': [[x, y], ...], # Tactical firefighter positions
                'threat_level': str,                    # 'low', 'medium', 'high'
                'analysis': str                         # Textual analysis
            }
        """
        pass

    def _validate_response(self, response: Dict) -> bool:
        required_fields = [
            'fire_locations',
            'fire_intensity',
            'wind_direction',
            'recommended_positions',
        ]
        return all(field in response for field in required_fields)

    def _create_default_response(self) -> Dict:
        return {
            'fire_locations': [],
            'fire_intensity': [],
            'wind_direction': 0.0,
            'recommended_positions': [],
            'threat_level': 'unknown',
            'analysis': 'VLM analysis failed',
        }
