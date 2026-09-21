"""Offline comparison of text-prompted segmentation; never connects to a robot.

Manifest CSV columns: image,prompt,scene_id,expected_tool. Scene images are
evaluation records only; no training or mask labels are used. Review the saved
overlays and fill in correctness by hand in a separate trial log.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ur5_grasp-main"))

from bsp.camera_bsp.sam_tool_detect import SamToolDetector


def main():
    parser = argparse.ArgumentParser(description="Compare local text-prompted tool segmenters")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--backend", choices=("clipseg_mobilesam", "yoloe"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("image") or not row.get("prompt") for row in rows):
        raise ValueError("manifest needs image and prompt on every row")
    detectors = {}
    review_rows = []
    log_path = args.output / "predictions.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for index, row in enumerate(rows, 1):
            image_path = (args.manifest.parent / row["image"]).resolve()
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(str(image_path))
            prompt = row["prompt"].strip()
            if prompt not in detectors:
                detectors[prompt] = SamToolDetector(
                    prompt, checkpoint=args.checkpoint, backend=args.backend)
            detector = detectors[prompt]
            # A uniform synthetic depth lets both backends use the same adapter.
            depth = image[:, :, 0].copy()
            depth.fill(1000)
            start = time.perf_counter()
            results = detector.detect_all(image, depth.astype("uint16"), 0.001)
            elapsed = time.perf_counter() - start
            best = results[0] if results else None
            inference = detector.last_inference
            overlay = detector.draw(image, best)
            overlay_name = f"{index:04d}.jpg"
            cv2.imwrite(str(args.output / overlay_name), overlay)
            log.write(json.dumps({
                "scene_id": row.get("scene_id", str(index)),
                "camera": row.get("camera", "unknown"),
                "image": str(image_path), "prompt": prompt,
                "expected_tool": row.get("expected_tool", ""),
                "backend": args.backend, "detections": len(results),
                "best_box": best["box"] if best else None,
                "best_score": best["score"] if best else None,
                "mask_area_px": best["area"] if best else None,
                "elapsed_s": elapsed, "overlay": overlay_name,
                "model_detection_s": inference.detection_seconds,
                "model_segmentation_s": inference.segmentation_seconds,
                "cuda_peak_allocated_mb": inference.cuda_peak_allocated_mb,
                "cuda_peak_reserved_mb": inference.cuda_peak_reserved_mb,
                "selected_correctly": None, "grasp_region_covered": None,
            }, ensure_ascii=False) + "\n")
            review_rows.append({
                "scene_id": row.get("scene_id", str(index)),
                "camera": row.get("camera", "unknown"),
                "expected_tool": row.get("expected_tool", ""),
                "prompt": prompt, "overlay": overlay_name,
                "selected_correctly": "", "grasp_region_covered": "",
                "notes": "",
            })
    with (args.output / "review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Saved {len(rows)} predictions to {log_path}")


if __name__ == "__main__":
    main()
