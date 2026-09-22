from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .preprocessing import SquareTransform


def decode_maps(outputs, transform: SquareTransform) -> dict[str, np.ndarray]:
    raw = [item[0, 0].detach().float().cpu().numpy() for item in outputs]
    expected = (transform.input_size, transform.input_size)
    if any(item.shape != expected or not np.isfinite(item).all() for item in raw):
        raise ValueError("model returned non-finite or unexpected output maps")
    quality, cosine, sine, width = raw
    quality = gaussian_filter(quality, 2.0, mode="nearest")
    cosine = gaussian_filter(cosine, 2.0, mode="nearest")
    sine = gaussian_filter(sine, 2.0, mode="nearest")
    # Training normalizes width by input_size / 2. The upstream inference helper
    # multiplies by 150, which does not match its 224-pixel training profile.
    width_factor = transform.input_size / 2.0
    width = gaussian_filter(width, 1.0, mode="nearest") * width_factor
    restored_cosine = transform.restore_map(cosine)
    restored_sine = transform.restore_map(sine)
    return {
        "quality": transform.restore_map(quality),
        "angle": np.arctan2(restored_sine, restored_cosine) / 2.0,
        "width": transform.restore_map(width) * transform.square_side / transform.input_size,
        "angle_strength": np.hypot(restored_sine, restored_cosine),
    }


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
