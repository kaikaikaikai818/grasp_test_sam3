from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GraspCandidate:
    center_xy: tuple[int, int]
    angle_rad: float
    angle_deg: float
    width_px: float
    quality: float
    angle_strength: float
    corners_xy: list[list[float]]


@dataclass(frozen=True)
class FusionResult:
    accepted: bool
    candidate: GraspCandidate | None
    rejection_reason: str | None

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "candidate": asdict(self.candidate) if self.candidate else None,
            "rejection_reason": self.rejection_reason,
        }


def select_masked_grasp(
    maps: dict[str, np.ndarray],
    mask: np.ndarray,
    min_quality: float = 0.05,
    min_angle_strength: float = 0.05,
    max_width_to_minor_bbox: float = 1.25,
    max_width_to_local_diameter: float = 1.5,
) -> FusionResult:
    quality, angle, width = (maps[key] for key in ("quality", "angle", "width"))
    strength = maps.get("angle_strength", np.ones_like(quality))
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != quality.shape:
        raise ValueError("mask and prediction map dimensions differ")
    if binary.sum() < 100:
        return FusionResult(False, None, "mask is empty or too small")
    valid = (
        binary & np.isfinite(quality) & np.isfinite(angle) & np.isfinite(width)
        & np.isfinite(strength) & (width > 0)
    )
    if not valid.any():
        return FusionResult(False, None, "mask contains no finite grasp geometry")
    row, column = np.unravel_index(np.argmax(np.where(valid, quality, -np.inf)), quality.shape)
    score = float(quality[row, column])
    if score < min_quality:
        return FusionResult(False, None, f"quality {score:.3f} is below {min_quality:.3f}")
    vector_strength = float(strength[row, column])
    if vector_strength < min_angle_strength:
        return FusionResult(False, None, "angle vector is too weak")
    x, y, box_width, box_height = cv2.boundingRect(binary.astype(np.uint8))
    predicted_width = float(width[row, column])
    width_limit = max_width_to_minor_bbox * min(box_width, box_height)
    if predicted_width > width_limit:
        return FusionResult(
            False, None,
            f"predicted width {predicted_width:.1f}px exceeds mask limit {width_limit:.1f}px",
        )
    distances = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    local_diameter = max(2.0 * float(distances[row, column]), 1.0)
    local_limit = max(8.0, max_width_to_local_diameter * local_diameter)
    if predicted_width > local_limit:
        return FusionResult(
            False, None,
            f"predicted width {predicted_width:.1f}px exceeds local target width limit {local_limit:.1f}px",
        )
    theta = float(angle[row, column])
    direction = np.array([np.cos(theta), -np.sin(theta)])
    perpendicular = np.array([np.sin(theta), np.cos(theta)])
    center = np.array([column, row], dtype=float)
    corners = np.array([
        center + u * predicted_width / 2 * direction + v * predicted_width / 4 * perpendicular
        for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ])
    candidate = GraspCandidate(
        (int(column), int(row)), theta, float(np.degrees(theta)), predicted_width,
        score, vector_strength, corners.tolist(),
    )
    return FusionResult(True, candidate, None)
