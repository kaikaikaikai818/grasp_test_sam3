from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class Position3D:
    xyz_m: tuple[float, float, float] | None
    valid_depth_points: int
    median_depth_m: float | None


def estimate_mask_position(
    mask: np.ndarray,
    depth_image: np.ndarray,
    depth_scale: float,
    intrinsics: Any,
    min_depth_m: float = 0.15,
    max_depth_m: float = 3.0,
    max_samples: int = 500,
) -> Position3D:
    if mask.shape != depth_image.shape:
        raise ValueError("mask 与对齐深度图尺寸不一致。")

    binary = mask.astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((5, 5), np.uint8), iterations=1)
    if not eroded.any():
        eroded = binary

    depth_m = depth_image.astype(np.float32) * float(depth_scale)
    valid = (eroded > 0) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    values = depth_m[valid]
    if values.size == 0:
        return Position3D(None, 0, None)

    median_depth = float(np.median(values))
    mad = float(np.median(np.abs(values - median_depth)))
    tolerance = max(0.03, 3.0 * 1.4826 * mad)
    valid &= np.abs(depth_m - median_depth) <= tolerance
    pixels_yx = np.argwhere(valid)
    valid_count = int(len(pixels_yx))
    if valid_count == 0:
        return Position3D(None, 0, median_depth)

    if valid_count > max_samples:
        indexes = np.linspace(0, valid_count - 1, max_samples, dtype=np.int32)
        pixels_yx = pixels_yx[indexes]

    points = []
    for y, x in pixels_yx:
        depth = float(depth_m[y, x])
        points.append(rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], depth))
    xyz = np.median(np.asarray(points, dtype=np.float32), axis=0)
    return Position3D(
        xyz_m=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        valid_depth_points=valid_count,
        median_depth_m=median_depth,
    )
