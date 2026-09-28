"""Adaptive height planning for a tape measure resting on a fixed surface."""
from __future__ import annotations

import numpy as np


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
