from __future__ import annotations

import cv2
import numpy as np


def heatmap(name: str, value: np.ndarray) -> np.ndarray:
    if name == "angle":
        display = (value + np.pi / 2) / np.pi
    elif name == "width":
        display = value / max(value.shape)
    else:
        display = value
    return cv2.applyColorMap(
        (np.clip(display, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
