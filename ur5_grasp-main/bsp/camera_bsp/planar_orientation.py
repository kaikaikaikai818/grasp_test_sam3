"""Plan an overhead gripper orientation from a segmented tabletop tool."""
from __future__ import annotations

import cv2
import numpy as np


def principal_axis_base(mask, depth_m, pixel_to_base, stride=4):
    """Return undirected tool-axis angle in base XY, or a rejection reason.

    Pixel samples use the measured depth at each point. This handles a wrist
    camera that is itself rotated relative to the robot base.
    """
    binary = np.asarray(mask, dtype=bool)
    depths = np.asarray(depth_m, dtype=np.float64)
    if binary.ndim != 2 or binary.shape != depths.shape:
        return None, "mask/depth dimensions differ"
    ys, xs = np.nonzero(binary[::stride, ::stride])
    points = []
    for u, v in zip(xs * stride, ys * stride):
        depth = float(depths[v, u])
        if not 0.15 <= depth <= 2.0:
            continue
        try:
            point = np.asarray(pixel_to_base(int(u), int(v), depth), dtype=float)
        except Exception:
            continue
        if point.size >= 2 and np.all(np.isfinite(point[:2])):
            points.append(point[:2])
    if len(points) < 20:
        return None, "too few 3-D mask samples for orientation"
    coords = np.asarray(points)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(coords, rowvar=False))
    if eigenvalues[0] <= 0 or eigenvalues[1] / eigenvalues[0] < 2.0:
        return None, "tool has no reliable planar long axis"
    axis = eigenvectors[:, 1]
    return float(np.arctan2(axis[1], axis[0]) % np.pi), None


def overhead_orientation(tool_axis_rad, current_rvec,
                         reference_rvec=(np.pi, 0.0, 0.0)):
    """Orient closing axis across tool axis; keep the reference downward tilt.

    The reference gripper closes along base X, as confirmed for this setup.
    Select the equivalent 180-degree gripper pose requiring less rotation.
    """
    if not np.isfinite(tool_axis_rad):
        raise ValueError("non-finite tool angle")
    reference, _ = cv2.Rodrigues(np.asarray(reference_rvec, dtype=float).reshape(3, 1))
    current, _ = cv2.Rodrigues(np.asarray(current_rvec, dtype=float).reshape(3, 1))
    candidates = []
    for yaw in (tool_axis_rad + np.pi / 2, tool_axis_rad - np.pi / 2):
        c, s = np.cos(yaw), np.sin(yaw)
        around_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        target = around_z @ reference
        delta = target @ current.T
        angular_distance = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
        rvec, _ = cv2.Rodrigues(target)
        candidates.append((float(angular_distance), rvec.reshape(3)))
    return min(candidates, key=lambda item: item[0])[1].tolist()


def axial_difference_deg(a, b):
    return float(np.degrees(abs((a - b + np.pi / 2) % np.pi - np.pi / 2)))
