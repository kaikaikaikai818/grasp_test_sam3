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
                    "rubber mallet"):
        return None, "no visual grasp strategy for this tool"
    distances = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
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
