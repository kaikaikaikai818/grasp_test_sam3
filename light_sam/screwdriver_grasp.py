"""Pure image/depth geometry for the screwdriver handle preview."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ScrewdriverGrasp:
    center_xy: tuple[int, int]
    angle_deg: float
    opening_width_px: float
    handle_radius_px: float
    valid_depth_points: int
    corners_xy: list[list[float]]

    def as_dict(self) -> dict:
        return asdict(self)


def plan_screwdriver_grasp(
    mask: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    min_radius_px: float = 4.0,
) -> tuple[ScrewdriverGrasp | None, str | None]:
    """Place a planar grasp on the thick handle, never on the thin shaft."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.shape != depth_raw.shape or not binary.any():
        return None, "invalid screwdriver mask or depth shape"
    distances = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    radius = float(distances.max())
    if radius < float(min_radius_px):
        return None, "screwdriver handle is too narrow"

    core = distances >= radius * 0.72
    component_count, component_labels, _, _ = cv2.connectedComponentsWithStats(
        core.astype(np.uint8)
    )
    if component_count <= 1:
        return None, "screwdriver handle interior unavailable"
    handle_label = max(
        range(1, component_count),
        key=lambda label: float(distances[component_labels == label].max()),
    )
    depth_m = np.asarray(depth_raw, dtype=np.float32) * float(depth_scale)
    valid = (
        (component_labels == handle_label) & np.isfinite(depth_m)
        & (depth_m >= 0.15) & (depth_m <= 2.0)
    )
    if int(valid.sum()) < 20:
        return None, "screwdriver handle has insufficient valid depth"
    valid_y, valid_x = np.nonzero(valid)
    center = np.array(
        (float(np.median(valid_x)), float(np.median(valid_y))), dtype=np.float64
    )

    mask_y, mask_x = np.nonzero(binary)
    points = np.column_stack((mask_x, mask_y)).astype(np.float64)
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    major /= max(float(np.linalg.norm(major)), 1e-9)
    closing = np.array((-major[1], major[0]), dtype=np.float64)
    opening = 2.0 * radius * 1.20
    jaw_depth = max(2.0 * radius, 8.0)
    corners = np.array([
        center + u * opening / 2.0 * closing + v * jaw_depth / 2.0 * major
        for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ])
    angle = float(np.degrees(np.arctan2(-closing[1], closing[0])))
    return ScrewdriverGrasp(
        center_xy=(int(round(center[0])), int(round(center[1]))),
        angle_deg=angle,
        opening_width_px=float(opening),
        handle_radius_px=radius,
        valid_depth_points=int(valid.sum()),
        corners_xy=corners.tolist(),
    ), None
