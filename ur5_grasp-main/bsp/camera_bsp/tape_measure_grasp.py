"""Adaptive height planning for a tape measure resting on a fixed surface."""
from __future__ import annotations

import numpy as np
import cv2


def isolate_tape_measure_body(mask):
    """Return the thick case core while discarding thin tape and wrist straps.

    The distance-transform maximum is inside the case.  Keeping its connected
    thick-core component removes attached narrow structures before estimating
    the case orientation.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.sum() < 100:
        return None, "invalid or too small tape-measure mask"
    distances = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    _, radius, _, (u, v) = cv2.minMaxLoc(distances)
    if radius < 5.0:
        return None, "tape-measure case has insufficient thick body area"
    core_threshold = max(4.0, float(radius) * 0.30)
    core = (distances >= core_threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(core, connectivity=8)
    if count <= 1:
        return None, "tape-measure body core is unavailable"
    label = int(labels[int(v), int(u)])
    if label <= 0:
        return None, "tape-measure body core does not contain its interior center"
    body = labels == label
    if int(body.sum()) < 100:
        return None, "tape-measure body core is too small"
    ys, xs = np.nonzero(body)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    if max(width, height) / max(1, min(width, height)) > 2.0:
        return None, "tape-measure candidate is strap-like, not case-like"
    return body, None


def result_at_tape_measure_body(result, depth_raw, depth_scale,
                                min_depth_m=0.15, max_depth_m=3.0):
    """Convert one segmentation into a case-only tracking result.

    This runs before temporal filtering so a high-score wrist strap cannot
    become the tracked target.  The original mask is retained only for support
    plane exclusion.
    """
    if result is None:
        return None, "tape-measure detection is unavailable"
    body, reason = isolate_tape_measure_body(result.get("mask"))
    if body is None:
        return None, reason
    depth_m = np.asarray(depth_raw, dtype=np.float32) * float(depth_scale)
    if depth_m.shape != body.shape:
        return None, "tape-measure mask/depth dimensions differ"
    valid = (body & np.isfinite(depth_m)
             & (depth_m >= float(min_depth_m))
             & (depth_m <= float(max_depth_m)))
    values = depth_m[valid]
    if values.size < 15:
        return None, "insufficient depth on tape-measure case"
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    valid &= np.abs(depth_m - median) <= max(0.03, 3.0 * 1.4826 * mad)
    pixels = np.argwhere(valid)
    if pixels.shape[0] < 15:
        return None, "insufficient consistent depth on tape-measure case"
    v, u = np.median(pixels, axis=0).astype(int)
    ys, xs = np.nonzero(body)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    contours, _ = cv2.findContours(
        body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle = 0.0
    if contours:
        (_, _), (width, height), raw_angle = cv2.minAreaRect(
            max(contours, key=cv2.contourArea))
        angle = float(raw_angle + (90.0 if width < height else 0.0)) % 180.0
    body_result = dict(result)
    body_result.update({
        "source_mask": np.asarray(result["mask"], dtype=bool),
        "mask": body,
        "center": (int(u), int(v)),
        "z_mm": median * 1000.0,
        "box": box,
        "area": float(body.sum()),
        "angle_deg": angle,
        "valid_depth_points": int(np.count_nonzero(valid)),
    })
    return body_result, None


def select_tape_measure_body_result(results, depth_raw, depth_scale):
    """Choose the best case-like result and reject strap-only segmentations."""
    valid = []
    reasons = []
    for result in results or ():
        body_result, reason = result_at_tape_measure_body(
            result, depth_raw, depth_scale)
        if body_result is not None:
            valid.append(body_result)
        elif reason:
            reasons.append(reason)
    if not valid:
        return None, (reasons[0] if reasons
                      else "no tape-measure case candidate")
    return max(valid, key=lambda item: (float(item.get("score", 0.0)),
                                        float(item.get("area", 0.0)))), None


def plan_tape_measure_grasp(body_top_z_m: float, support_plane_z_m: float,
                            gripper_offset_m: float = 0.0,
                            center_bias_m: float = 0.0,
                            minimum_clearance_m: float = 0.008,
                            minimum_thickness_m: float = 0.015,
                            maximum_thickness_m: float = 0.100):
    """Return a TCP height at the tape-measure body's vertical midpoint.

    ``body_top_z_m`` is the D435i base-frame depth at the interior body
    candidate.  The fixed support plane is shared by every tool.  The body
    thickness is therefore measured on every attempt instead of stored as a
    per-tool calibration value.
    """
    values = np.asarray([
        body_top_z_m, support_plane_z_m, gripper_offset_m, center_bias_m,
        minimum_clearance_m, minimum_thickness_m, maximum_thickness_m,
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None, "non-finite tape-measure height or support plane"
    if minimum_thickness_m <= 0 or maximum_thickness_m <= minimum_thickness_m:
        return None, "invalid tape-measure thickness limits"

    thickness = float(body_top_z_m) - float(support_plane_z_m)
    if thickness < float(minimum_thickness_m):
        return None, "tape-measure body thickness is implausibly small"
    if thickness > float(maximum_thickness_m):
        return None, "tape-measure body thickness is implausibly large"

    body_mid_z = float(support_plane_z_m) + thickness / 2.0
    nominal_tcp_z = body_mid_z + float(gripper_offset_m)
    requested_tcp_z = nominal_tcp_z + float(center_bias_m)
    minimum_tcp_z = float(support_plane_z_m) + max(0.0, float(minimum_clearance_m))
    tcp_z = max(requested_tcp_z, minimum_tcp_z)
    return tcp_z, {
        "tool_category": "tape measure",
        "body_top_z_m": float(body_top_z_m),
        "body_thickness_m": thickness,
        "body_mid_z_m": body_mid_z,
        "support_plane_z_m": float(support_plane_z_m),
        "gripper_offset_m": float(gripper_offset_m),
        "nominal_grasp_tcp_z_m": nominal_tcp_z,
        "requested_center_bias_m": float(center_bias_m),
        "applied_center_bias_m": tcp_z - nominal_tcp_z,
        "minimum_clearance_m": max(0.0, float(minimum_clearance_m)),
        "grasp_tcp_z_m": tcp_z,
    }
