"""Depth geometry helpers for the first, handle-only screwdriver grasp.

The helpers are deliberately free of robot I/O so they can be tested without
camera or robot hardware.  They return a reason rather than guessing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class HandleCandidate:
    center_px: tuple[int, int]
    depth_m: float
    radius_px: float
    valid_depth_points: int


@dataclass(frozen=True)
class SupportPlane:
    z_m: float
    spread_m: float
    inlier_count: int


def find_screwdriver_handle(result: dict, depth_raw: np.ndarray, depth_scale: float,
                            min_radius_px: float = 4.0) -> tuple[Optional[HandleCandidate], str]:
    """Find the thickest interior mask region, which is the screwdriver handle.

    A distance transform favours the broad handle over the narrow metal shaft.
    Depth is sampled only from a compact region around the selected interior.
    """
    mask = np.asarray(result.get("mask"), dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return None, "empty screwdriver mask"
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radius = float(distances.max())
    if radius < min_radius_px:
        return None, "screwdriver handle is too narrow"
    core = distances >= radius * 0.72
    components, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8))
    if components <= 1:
        return None, "handle interior unavailable"
    # Use the component with the largest maximum distance, then its geometric median.
    best_label = max(range(1, components), key=lambda label: float(distances[labels == label].max()))
    ys, xs = np.nonzero(labels == best_label)
    depth_m = np.asarray(depth_raw, dtype=np.float32) * float(depth_scale)
    valid = ((labels == best_label) & np.isfinite(depth_m) &
             (depth_m >= 0.15) & (depth_m <= 2.0))
    if int(valid.sum()) < 20:
        return None, "handle has insufficient valid depth"
    vy, vx = np.nonzero(valid)
    return HandleCandidate(
        center_px=(int(round(float(np.median(vx)))), int(round(float(np.median(vy))))),
        depth_m=float(np.median(depth_m[valid])),
        radius_px=radius,
        valid_depth_points=int(valid.sum()),
    ), "thickest screwdriver handle region"


def result_at_handle(result: dict, handle: HandleCandidate) -> dict:
    """Preserve detector metadata while using the safe handle point for 3-D work."""
    selected = dict(result)
    selected["center"] = handle.center_px
    selected["z_mm"] = handle.depth_m * 1000.0
    selected["valid_depth_points"] = handle.valid_depth_points
    selected["handle_radius_px"] = handle.radius_px
    return selected


def fit_horizontal_support_plane(depth_raw: np.ndarray, depth_scale: float,
                                 pixel_to_base: Callable[[int, int, float], np.ndarray],
                                 exclude_mask: Optional[np.ndarray] = None,
                                 stride: int = 6) -> tuple[Optional[SupportPlane], str]:
    """Robustly fit the dominant horizontal base-frame plane from depth samples."""
    depth_m = np.asarray(depth_raw, dtype=np.float32) * float(depth_scale)
    height, width = depth_m.shape[:2]
    excluded = (np.zeros_like(depth_m, dtype=bool) if exclude_mask is None
                else np.asarray(exclude_mask, dtype=bool))
    zs = []
    for v in range(0, height, int(stride)):
        for u in range(0, width, int(stride)):
            z = float(depth_m[v, u])
            if excluded[v, u] or not (0.15 <= z <= 2.0):
                continue
            try:
                point = np.asarray(pixel_to_base(u, v, z), dtype=float).reshape(3)
            except Exception:
                continue
            if np.all(np.isfinite(point)):
                zs.append(float(point[2]))
    if len(zs) < 80:
        return None, "too few depth samples for support plane"
    values = np.asarray(zs, dtype=np.float64)
    # A 3 mm histogram identifies the largest horizontal surface without
    # assuming that every visible pixel belongs to the cardboard.
    bins = np.round(values / 0.003).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    dominant = unique[int(np.argmax(counts))] * 0.003
    inliers = values[np.abs(values - dominant) <= 0.006]
    if inliers.size < 60:
        return None, "support plane has too few inliers"
    z_m = float(np.median(inliers))
    spread = float(np.max(np.abs(inliers - z_m)))
    if spread > 0.008:
        return None, "support plane depth spread is too large"
    return SupportPlane(z_m=z_m, spread_m=spread, inlier_count=int(inliers.size)), "support plane fitted"


def save_calibration(path: Path, calibration: dict) -> None:
    path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"support_plane_z_m", "support_plane_spread_m", "contact_offset_m",
                "minimum_tcp_plane_clearance_m", "handle_surface_above_plane_m"}
    if not required.issubset(data):
        raise ValueError("grasp calibration missing required fields")
    return data


def plane_to_dict(plane: SupportPlane) -> dict:
    return asdict(plane)
