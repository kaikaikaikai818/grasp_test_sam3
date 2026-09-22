from __future__ import annotations

import cv2
import numpy as np

from .fusion import FusionResult, select_masked_grasp
from .types import PredictionResult


def _principal_axis(mask: np.ndarray) -> tuple[np.ndarray, float]:
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64)
    if len(points) < 2:
        return np.array((1.0, 0.0)), 0.0
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    major = vectors[:, order[-1]]
    major /= max(np.linalg.norm(major), 1e-9)
    major_value = max(float(values[order[-1]]), 1e-9)
    minor_value = max(float(values[order[-2]]), 0.0)
    anisotropy = float(np.clip(1.0 - minor_value / major_value, 0.0, 1.0))
    return major, anisotropy


def _choose_center(
    mask: np.ndarray, distance: np.ndarray, major: np.ndarray
) -> tuple[tuple[int, int], np.ndarray]:
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64)
    projection = points @ major
    low, high = np.quantile(projection, (0.05, 0.95))
    if high - low < 2:
        index = int(np.argmax(distance[ys, xs]))
        return (int(xs[index]), int(ys[index])), major

    edges = np.linspace(low, high, 22)
    centers: list[tuple[int, int]] = []
    radii: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        selected = (projection >= start) & (projection < end)
        if not selected.any():
            centers.append((-1, -1))
            radii.append(0.0)
            continue
        selected_indices = np.flatnonzero(selected)
        local_distances = distance[ys[selected], xs[selected]]
        point_index = selected_indices[int(np.argmax(local_distances))]
        centers.append((int(xs[point_index]), int(ys[point_index])))
        radii.append(float(distance[ys[point_index], xs[point_index]]))

    valid_indices = [index for index in range(4, 17) if radii[index] > 0]
    if not valid_indices:
        index = int(np.argmax(distance[ys, xs]))
        return (int(xs[index]), int(ys[index])), major

    # Use a three-slice median so a small notch or mask defect cannot become
    # the grasp site. Among similarly thin slices, prefer the object middle.
    smoothed = {}
    for index in valid_indices:
        neighborhood = [radii[item] for item in range(max(0, index - 1), min(21, index + 2))
                        if radii[item] > 0]
        smoothed[index] = float(np.median(neighborhood))
    minimum = min(smoothed.values())
    comparable = [index for index in valid_indices if smoothed[index] <= minimum * 1.15]
    chosen = min(comparable, key=lambda index: abs(index - 10))

    # Re-estimate orientation from a local slab around the selected handle
    # segment. A large wrench head should not rotate the closing direction.
    slab_low = edges[max(0, chosen - 2)]
    slab_high = edges[min(21, chosen + 3)]
    slab_points = points[(projection >= slab_low) & (projection < slab_high)]
    local_major = major
    if len(slab_points) >= 20:
        values, vectors = np.linalg.eigh(np.cov(slab_points, rowvar=False))
        local_major = vectors[:, int(np.argmax(values))]
        local_major /= max(np.linalg.norm(local_major), 1e-9)
    return centers[chosen], local_major


def geometry_maps(mask: np.ndarray, opening_scale: float = 1.15) -> dict[str, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    shape = binary.shape
    quality = np.zeros(shape, np.float32)
    angle = np.zeros(shape, np.float32)
    width = np.zeros(shape, np.float32)
    strength = np.zeros(shape, np.float32)
    if binary.sum() == 0:
        return {"quality": quality, "angle": angle, "width": width,
                "angle_strength": strength}

    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    major, anisotropy = _principal_axis(binary)
    (center_x, center_y), local_major = _choose_center(binary, distance, major)

    closing = np.array((-local_major[1], local_major[0]))
    theta = np.arctan2(-closing[1], closing[0])
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    local_radius = max(float(distance[center_y, center_x]), 1.0)
    sigma = max(local_radius * 1.5, 4.0)
    grid_y, grid_x = np.indices(shape)
    confidence = float(0.60 + 0.30 * anisotropy)
    quality[:] = (
        confidence
        * np.exp(-((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2 * sigma**2))
        * binary
    )
    angle[binary] = theta
    width[binary] = 2.0 * distance[binary] * float(opening_scale)
    strength[binary] = max(0.10, anisotropy)
    return {"quality": quality, "angle": angle, "width": width,
            "angle_strength": strength}


class GeometryGraspPredictor:
    backend_name = "mask geometry baseline"

    def __init__(self, opening_scale: float = 1.15):
        if not 1.0 <= opening_scale <= 1.45:
            raise ValueError("opening_scale must be between 1.0 and 1.45")
        self.opening_scale = float(opening_scale)

    def predict(self, image, text: str, masks) -> PredictionResult:
        height, width = image.shape[:2]
        combined = {
            "quality": np.zeros((height, width), np.float32),
            "angle": np.zeros((height, width), np.float32),
            "width": np.zeros((height, width), np.float32),
            "angle_strength": np.zeros((height, width), np.float32),
        }
        grasps: list[FusionResult] = []
        for raw_mask in masks:
            mask = np.asarray(raw_mask, dtype=bool)
            if mask.shape != (height, width):
                raise ValueError("mask and image dimensions differ")
            maps = geometry_maps(mask, self.opening_scale)
            result = select_masked_grasp(maps, mask)
            grasps.append(result)
            replace = maps["quality"] > combined["quality"]
            for name in combined:
                combined[name][replace] = maps[name][replace]
        return PredictionResult(
            combined,
            grasps,
            {"type": "mask_geometry", "image_height": height, "image_width": width},
            self.backend_name,
        )
