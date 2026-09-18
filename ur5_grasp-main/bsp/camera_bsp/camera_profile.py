"""Small, hardware-free checks for an expected RealSense stream profile."""
from __future__ import annotations

import numpy as np


def max_intrinsics_delta(live: np.ndarray, expected: np.ndarray) -> float:
    """Return the largest fx, fy, cx or cy difference in pixels.

    Invalid input returns infinity so the caller always rejects it.
    """
    live = np.asarray(live, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if live.shape != (3, 3) or expected.shape != (3, 3):
        return float("inf")
    if not np.all(np.isfinite(live)) or not np.all(np.isfinite(expected)):
        return float("inf")
    return float(max(abs(live[0, 0] - expected[0, 0]),
                     abs(live[1, 1] - expected[1, 1]),
                     abs(live[0, 2] - expected[0, 2]),
                     abs(live[1, 2] - expected[1, 2])))
