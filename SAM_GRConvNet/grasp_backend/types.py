from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fusion import FusionResult


@dataclass(frozen=True)
class PredictionResult:
    maps: dict[str, np.ndarray]
    grasps: list[FusionResult]
    transform: dict[str, object]
    backend: str
