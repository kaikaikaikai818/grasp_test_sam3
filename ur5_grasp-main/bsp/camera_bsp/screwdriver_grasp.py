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
    depth_m: float            # robust handle-top depth (closest surface)
    median_depth_m: float     # median depth, kept for diagnostics
    radius_px: float
    valid_depth_points: int


@dataclass(frozen=True)
class SupportPlane:
    z_m: float
    spread_m: float
    inlier_count: int


def base_point_m(camera_to_base_result) -> np.ndarray:
    """Extract the metre-valued point returned by ``camera_to_base``.

    ``UR_Robot.camera_to_base`` returns ``(point_mm, point_m)``.  Centralising
    the unpacking prevents an already-metre value from being divided by 1000
    a second time.
    """
    if not isinstance(camera_to_base_result, tuple) or len(camera_to_base_result) != 2:
        raise ValueError("camera_to_base must return (point_mm, point_m)")
    point = np.asarray(camera_to_base_result[1], dtype=np.float64).reshape(-1)
    if point.size != 3 or not np.all(np.isfinite(point)):
        raise ValueError("invalid base-frame point in metres")
    return point


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
    depths = np.asarray(depth_m[valid], dtype=np.float64)
    # The handle top is the surface closest to a downward-looking camera, so a
    # low percentile is a robust estimate of the top rather than the median,
    # which is dragged towards the sloping sides and the surrounding plane.
    top_depth = float(np.percentile(depths, 10.0))
    median_depth = float(np.median(depths))
    return HandleCandidate(
        center_px=(int(round(float(np.median(vx)))), int(round(float(np.median(vy))))),
        depth_m=top_depth,
        median_depth_m=median_depth,
        radius_px=radius,
        valid_depth_points=int(valid.sum()),
    ), None


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


def estimate_handle_thickness(radius_px: float, depth_m: float,
                              focal_px: float) -> tuple[Optional[float], str]:
    """Estimate the handle diameter from its projected width and distance.

    A cylinder's silhouette width equals its diameter regardless of viewing
    angle, so ``diameter = 2 * radius_px * depth / focal``.  This avoids relying
    on the depth of a dark, IR-absorbing handle surface, which reads as the
    surrounding plane.  Returns ``(thickness_m, reason)``.
    """
    values = np.asarray([radius_px, depth_m, focal_px], dtype=np.float64)
    if not np.all(np.isfinite(values)) or focal_px <= 0.0 or depth_m <= 0.0:
        return None, "invalid radius, depth, or focal length"
    diameter = 2.0 * float(radius_px) * float(depth_m) / float(focal_px)
    return float(diameter), "handle diameter from projected width"


def plan_grasp_tcp(handle_thickness_m: float, support_plane_z_m: float,
                   gripper_offset_m: float):
    """Grasp the handle at its mid height using a tool-independent gripper offset.

    ``gripper_offset_m`` is the constant vertical distance from the grasp point
    (handle mid) to the active TCP, measured once by the operator.  The handle
    thickness is re-estimated every attempt, so a different screwdriver does not
    need a new calibration.  Returns ``(tcp_z, plan)`` or ``(None, reason)``.
    """
    values = np.asarray([handle_thickness_m, support_plane_z_m, gripper_offset_m], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None, "non-finite handle thickness, support plane, or gripper offset"
    thickness = float(handle_thickness_m)
    if thickness <= 0.010:
        return None, "handle thickness is implausibly small"
    if thickness > 0.100:
        return None, "handle thickness is implausibly large"
    handle_mid_z = float(support_plane_z_m) + thickness / 2.0
    tcp_z = handle_mid_z + float(gripper_offset_m)
    return tcp_z, {
        "handle_thickness_m": thickness,
        "handle_mid_z_m": handle_mid_z,
        "gripper_offset_m": float(gripper_offset_m),
        "grasp_tcp_z_m": tcp_z,
    }


def save_calibration(path: Path, calibration: dict) -> None:
    path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path, default_gripper_offset_m: Optional[float] = None) -> Optional[dict]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"support_plane_z_m", "support_plane_spread_m"}
    if not required.issubset(data):
        raise ValueError("support-plane calibration missing required fields")
    plane_z = float(data["support_plane_z_m"])
    plane_spread = float(data["support_plane_spread_m"])
    if not np.isfinite(plane_z) or not np.isfinite(plane_spread):
        raise ValueError("support-plane calibration contains non-finite values")
    if not (-0.2 <= plane_z <= 0.6):
        raise ValueError("support-plane calibration is outside the robot workspace")
    if not (0.0 <= plane_spread <= 0.008):
        raise ValueError("support-plane calibration spread exceeds 8 mm")
    if default_gripper_offset_m is not None:
        data["gripper_offset_m"] = float(default_gripper_offset_m)
        data["gripper_offset_source"] = "active_tcp_clamp"
    elif "gripper_offset_m" not in data:
        raise ValueError("gripper offset is missing and no TCP-clamp default was provided")
    if not np.isfinite(float(data["gripper_offset_m"])):
        raise ValueError("gripper offset is not finite")
    return data


def verified_support_plane_z(live_plane: Optional[SupportPlane], calibration: Optional[dict],
                             max_shift_m: float = 0.008):
    """Use the saved fixed plane only when live D435i depth agrees with it."""
    if live_plane is None:
        return None, "live support plane unavailable"
    if calibration is None or "support_plane_z_m" not in calibration:
        return None, "fixed support plane is not calibrated"
    reference = float(calibration["support_plane_z_m"])
    delta = abs(float(live_plane.z_m) - reference)
    if delta > float(max_shift_m):
        return None, "live support plane differs from saved plane by %.1fmm" % (delta * 1000.0)
    return reference, None


def plane_to_dict(plane: SupportPlane) -> dict:
    return asdict(plane)
