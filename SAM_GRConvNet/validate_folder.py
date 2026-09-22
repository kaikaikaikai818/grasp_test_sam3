"""Evaluate original/90-degree rotation consistency for a folder of images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app import build_models, infer, load_config


def axial_error_deg(actual: float, expected: float) -> float:
    return float(np.degrees(abs((actual - expected + np.pi / 2) % np.pi - np.pi / 2)))


def first_candidate(prediction):
    if prediction:
        for item in prediction.grasps:
            if item.accepted:
                return item.candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/validation.json"))
    args = parser.parse_args()
    paths = sorted({p for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                    for p in args.images.glob(pattern)})
    if len(paths) < 10:
        raise ValueError(f"at least 10 images are required; found {len(paths)}")
    sam, grasp = build_models(load_config(args.config))
    rows = []
    for path in paths:
        image = cv2.imread(str(path))
        objects, prediction = infer(image, args.text, sam, grasp)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rot_objects, rot_prediction = infer(rotated, args.text, sam, grasp)
        first, second = first_candidate(prediction), first_candidate(rot_prediction)
        error = None
        if first and second:
            # In image coordinates a clockwise image rotation subtracts pi/2.
            error = axial_error_deg(second.angle_rad, first.angle_rad - np.pi / 2)
        rows.append({
            "image": str(path), "accepted": first is not None,
            "rotated_accepted": second is not None, "rotation_error_deg": error,
            "objects": len(objects), "rotated_objects": len(rot_objects),
        })
        print(f"{path.name}: accepted={first is not None}, rotation_error={error}")
    errors = [row["rotation_error_deg"] for row in rows if row["rotation_error_deg"] is not None]
    summary = {
        "image_count": len(rows),
        "accepted_pair_count": len(errors),
        "median_rotation_error_deg": float(np.median(errors)) if errors else None,
        "passes_rotation_gate": bool(len(errors) == len(rows) and np.median(errors) <= 15.0),
        "robot_motion_authorized": False,
        "items": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "items"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
