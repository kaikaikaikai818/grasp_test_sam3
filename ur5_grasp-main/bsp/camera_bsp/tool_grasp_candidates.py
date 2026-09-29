"""Conservative 2-D grasp-region proposals for non-screwdriver tools.

These are visual candidates, not robot-ready TCP heights. Physical clearance,
finger width and the support plane must be established before actuation.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GraspCandidate:
    center_px: tuple[int, int]
    depth_m: float
    region_radius_px: float
    reason: str


def propose_grasp_region(mask, depth_raw, depth_scale, tool):
    binary = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_raw, dtype=float) * float(depth_scale)
    if binary.ndim != 2 or binary.shape != depth.shape or binary.sum() < 100:
        return None, "invalid or too small tool mask"
    if tool not in ("adjustable wrench", "tape measure", "tape dispenser",
                    "pliers"):
        return None, "no visual grasp strategy for this tool"
    distances = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    if tool == "pliers":
        return _propose_pliers_handles(binary, depth)
    if tool == "tape measure":
        _, radius, _, (u, v) = cv2.minMaxLoc(distances)
        if radius < 5:
            return None, "tape measure has insufficient body area"
    else:
        ys, xs = np.nonzero(binary)
        coords = np.column_stack((xs, ys)).astype(float)
        values, vectors = np.linalg.eigh(np.cov(coords, rowvar=False))
        if values[0] <= 0 or values[1] / values[0] < 2.0:
            return None, "handle direction is not distinguishable"
        axis = vectors[:, 1]
        along = (coords - coords.mean(axis=0)) @ axis
        minimum, maximum = np.percentile(along, [2, 98])
        if maximum - minimum < 25:
            return None, "handle is too short in image"
        outer_width = []
        for side in (along < minimum + .15 * (maximum - minimum),
                     along > maximum - .15 * (maximum - minimum)):
            outer_width.append(float(np.percentile(distances[ys[side], xs[side]], 85))
                               if np.any(side) else 0.0)
        if max(outer_width) < 4 or abs(outer_width[0] - outer_width[1]) < 2:
            return None, "head and handle ends are ambiguous"
        head_is_low = outer_width[0] > outer_width[1]
        fraction = (along - minimum) / (maximum - minimum)
        handle_band = ((fraction >= .55) & (fraction <= .80) if head_is_low
                       else (fraction >= .20) & (fraction <= .45))
        region = np.zeros_like(binary)
        region[ys[handle_band], xs[handle_band]] = True
        if not region.any():
            return None, "no safe handle band"
        region_distances = np.where(region, distances, 0)
        _, radius, _, (u, v) = cv2.minMaxLoc(region_distances)
        if radius < 4:
            return None, "handle band is too narrow"
    yy, xx = np.ogrid[:binary.shape[0], :binary.shape[1]]
    nearby = (xx - u) ** 2 + (yy - v) ** 2 <= max(radius * .6, 3) ** 2
    samples = depth[binary & nearby & np.isfinite(depth) & (depth >= .15) & (depth <= 2)]
    if samples.size < 15:
        return None, "insufficient depth at proposed grasp region"
    return GraspCandidate((int(u), int(v)), float(np.median(samples)),
                          float(radius), "interior handle/body candidate"), None


def _propose_pliers_handles(binary, depth):
    """Place the grasp center between the two handles, away from jaws/pivot."""
    ys, xs = np.nonzero(binary)
    coords = np.column_stack((xs, ys)).astype(float)
    center = coords.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(coords, rowvar=False))
    if values[0] <= 0 or values[1] / values[0] < 2.0:
        return None, "pliers long axis is not distinguishable"
    axis = vectors[:, 1]
    lateral_axis = np.asarray([-axis[1], axis[0]])
    relative = coords - center
    along = relative @ axis
    lateral = relative @ lateral_axis
    minimum, maximum = np.percentile(along, [2, 98])
    length = float(maximum - minimum)
    if length < 35:
        return None, "pliers are too short in image"

    end_masks = (along <= minimum + 0.25 * length,
                 along >= maximum - 0.25 * length)
    spans = []
    for end in end_masks:
        if np.count_nonzero(end) < 30:
            spans.append(0.0)
        else:
            low, high = np.percentile(lateral[end], [5, 95])
            spans.append(float(high - low))
    handle_is_low = spans[0] > spans[1]
    handle_span = max(spans)
    if handle_span < 12 or max(spans) < min(spans) * 1.20:
        return None, "pliers handle end is not distinct from jaws"

    target_along = (minimum + 0.28 * length if handle_is_low
                    else maximum - 0.28 * length)
    band = np.abs(along - target_along) <= 0.08 * length
    if np.count_nonzero(band) < 30:
        return None, "pliers handle band is incomplete"
    low, high = np.percentile(lateral[band], [5, 95])
    target_lateral = float((low + high) / 2.0)
    point = center + axis * target_along + lateral_axis * target_lateral
    u, v = int(round(point[0])), int(round(point[1]))
    if not (0 <= u < binary.shape[1] and 0 <= v < binary.shape[0]):
        return None, "pliers handle center lies outside image"

    samples = depth[ys[band], xs[band]]
    samples = samples[np.isfinite(samples) & (samples >= .15) & (samples <= 2)]
    if samples.size < 15:
        return None, "insufficient depth on pliers handles"
    return GraspCandidate(
        (u, v), float(np.median(samples)), handle_span / 2.0,
        "center between pliers handle pair"), None
