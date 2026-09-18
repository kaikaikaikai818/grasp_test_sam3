"""Geometry-only grasp planning for a text-segmented, single tabletop tool.

This module deliberately knows nothing about object names.  A prompt selects an
instance; the mask and depth decide whether it provides a safe parallel-jaw
candidate.  It returns a rejection reason instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class PixelGraspCandidate:
    center_px: tuple[int, int]
    depth_m: float
    axis_start_px: tuple[int, int]
    axis_end_px: tuple[int, int]
    width_px: float
    aspect_ratio: float


@dataclass(frozen=True)
class BaseGraspPlan:
    target_base_xyz_m: tuple[float, float, float]
    axis_base_xy: tuple[float, float]
    jaw_width_m: float
    candidate: PixelGraspCandidate


def select_one(results: list[dict], minimum_score: float) -> tuple[Optional[dict], str]:
    """Require exactly one usable text-prompt instance; never choose arbitrarily."""
    usable = [item for item in results if item.get("z_mm") is not None
              and float(item.get("score", 0.0)) >= minimum_score]
    if not usable:
        return None, "no usable prompt instance"
    if len(usable) != 1:
        return None, "multiple prompt instances (%d)" % len(usable)
    return usable[0], "single prompt instance"


def pixel_grasp_candidate(result: dict, depth_raw: np.ndarray, depth_scale: float,
                          min_aspect_ratio: float = 1.35,
                          min_pixels: int = 120) -> tuple[Optional[PixelGraspCandidate], str]:
    """Choose a conservative central pinch band from one stable segmentation mask.

    The candidate comes from the middle 42% of the principal axis.  This avoids
    a tool's tips and large end features without embedding a wrench/screwdriver
    specific rule.  Low-aspect or sparse masks are rejected for this first
    parallel-jaw implementation.
    """
    mask = np.asarray(result.get("mask"), dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return None, "empty mask"
    depth_m = np.asarray(depth_raw, dtype=np.float32) * float(depth_scale)
    valid = mask & np.isfinite(depth_m) & (depth_m >= 0.15) & (depth_m <= 3.0)
    ys, xs = np.nonzero(valid)
    if xs.size < min_pixels:
        return None, "too few valid depth pixels"

    pixels = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    center = np.median(pixels, axis=0)
    centered = pixels - center
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    if singular.size < 2 or singular[1] <= 1e-6:
        return None, "degenerate mask"
    aspect = float(singular[0] / singular[1])
    if aspect < min_aspect_ratio:
        return None, "mask is not an elongated graspable shape"
    axis = vectors[0]
    side = vectors[1]
    s = centered @ axis
    t = centered @ side
    extent = float(np.percentile(s, 95) - np.percentile(s, 5))
    if extent < 18.0:
        return None, "mask is too short"
    middle = np.abs(s - np.median(s)) <= extent * 0.21
    if int(np.count_nonzero(middle)) < max(40, min_pixels // 4):
        return None, "central pinch band is too small"

    central_pixels = pixels[middle]
    central_depth = depth_m[ys[middle], xs[middle]]
    pinch_center = np.median(central_pixels, axis=0)
    z = float(np.median(central_depth))
    width_px = float(np.percentile(t[middle], 90) - np.percentile(t[middle], 10))
    start = pinch_center - axis * extent * 0.30
    end = pinch_center + axis * extent * 0.30
    return PixelGraspCandidate(
        center_px=(int(round(pinch_center[0])), int(round(pinch_center[1]))),
        depth_m=z,
        axis_start_px=(int(round(start[0])), int(round(start[1]))),
        axis_end_px=(int(round(end[0])), int(round(end[1]))),
        width_px=width_px,
        aspect_ratio=aspect,
    ), "central pinch band"


def base_grasp_plan(candidate: PixelGraspCandidate, intrinsics: np.ndarray,
                    pixel_to_base: Callable[[int, int, float], np.ndarray],
                    min_jaw_width_m: Optional[float], max_jaw_width_m: Optional[float]) -> tuple[Optional[BaseGraspPlan], str]:
    """Convert a pixel candidate into a base-frame target and horizontal axis."""
    center = np.asarray(pixel_to_base(*candidate.center_px, candidate.depth_m), dtype=np.float64).reshape(3)
    start = np.asarray(pixel_to_base(*candidate.axis_start_px, candidate.depth_m), dtype=np.float64).reshape(3)
    end = np.asarray(pixel_to_base(*candidate.axis_end_px, candidate.depth_m), dtype=np.float64).reshape(3)
    direction = end[:2] - start[:2]
    length = float(np.linalg.norm(direction))
    if length < 0.005:
        return None, "3D tool axis is too short"
    direction /= length
    fx = float(np.asarray(intrinsics)[0, 0])
    jaw_width = candidate.width_px * candidate.depth_m / fx
    if min_jaw_width_m is not None and max_jaw_width_m is not None and not (
            min_jaw_width_m <= jaw_width <= max_jaw_width_m):
        return None, "candidate width %.1fmm outside gripper range" % (jaw_width * 1000.0)
    return BaseGraspPlan(
        target_base_xyz_m=tuple(float(value) for value in center),
        axis_base_xy=tuple(float(value) for value in direction),
        jaw_width_m=float(jaw_width),
        candidate=candidate,
    ), "geometric parallel-jaw candidate"
