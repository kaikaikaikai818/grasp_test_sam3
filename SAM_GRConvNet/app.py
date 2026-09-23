from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from grasp_backend import GeometryGraspPredictor
from grasp_backend.fusion import FusionResult
from grasp_backend.postprocessing import heatmap
from grasp_backend.types import PredictionResult
from segmentation import SamPipeline


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    for section, key in (("sam", "checkpoint"), ("sam", "clipseg_model")):
        value = Path(config[section][key])
        if not value.is_absolute():
            value = (config_path.parent / value).resolve()
        config[section][key] = str(value)
    return config


def build_models(config: dict):
    sam_cfg, grasp_cfg = config["sam"], config["grasp"]
    recognition_cfg = config.get("recognition", {})
    sam = SamPipeline(
        sam_cfg["checkpoint"], sam_cfg["clipseg_model"],
        device=sam_cfg.get("device", "cuda"),
        threshold=sam_cfg.get("threshold", 0.3),
        min_area=sam_cfg.get("min_area", 100),
        min_score=sam_cfg.get("min_score", 0.35),
        max_objects=sam_cfg.get("max_objects", 10),
        complete_object=sam_cfg.get("complete_object", True),
        sam_score_tolerance=sam_cfg.get("sam_score_tolerance", 0.12),
        max_mask_fraction=sam_cfg.get("max_mask_fraction", 0.70),
        overlap_merge=sam_cfg.get("overlap_merge", 0.20),
        categories=recognition_cfg.get("categories"),
        semantic_min_similarity=recognition_cfg.get("min_similarity", 0.20),
        semantic_min_margin=recognition_cfg.get("min_margin", 0.01),
        crop_padding=recognition_cfg.get("crop_padding", 0.10),
    )
    backend = grasp_cfg.get("backend", "geometry")
    if backend != "geometry":
        raise ValueError(f"unsupported grasp backend: {backend}")
    grasp = GeometryGraspPredictor(grasp_cfg.get("opening_scale", 1.15))
    return sam, grasp


def infer(image: np.ndarray, prompt: str, sam, grasp):
    objects = sam.predict(image, prompt)
    if not objects:
        return objects, None
    raw_prediction = grasp.predict(image, prompt, [item.mask for item in objects])
    filtered_grasps = [
        result if item.semantic_accepted else FusionResult(False, None, item.rejection_reason)
        for item, result in zip(objects, raw_prediction.grasps)
    ]
    prediction = PredictionResult(
        raw_prediction.maps, filtered_grasps,
        raw_prediction.transform, raw_prediction.backend,
    )
    return objects, prediction


def render(image: np.ndarray, objects, prediction):
    display = image.copy()
    for index, obj in enumerate(objects, 1):
        color = np.array((0, 190, 0) if obj.semantic_accepted else (0, 0, 210), dtype=np.uint8)
        display[obj.mask] = (display[obj.mask] * 0.65 + color * 0.35).astype(np.uint8)
        x1, y1, x2, y2 = obj.box
        box_color = (0, 200, 0) if obj.semantic_accepted else (0, 0, 230)
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
        fusion = prediction.grasps[index - 1] if prediction else None
        if fusion and fusion.accepted:
            candidate = fusion.candidate
            corners = np.rint(candidate.corners_xy).astype(np.int32)
            cv2.polylines(display, [corners], True, (0, 230, 255), 3)
            cv2.circle(display, candidate.center_xy, 6, (0, 0, 255), -1)
            label = (f"#{index} {obj.target_class} C={obj.target_score:.3f} "
                     f"Q={candidate.quality:.3f} A={candidate.angle_deg:.1f} W={candidate.width_px:.0f}px")
        else:
            reason = fusion.rejection_reason if fusion else "no grasp prediction"
            label = f"#{index} REJECT: {reason}"
        cv2.putText(display, label, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)
    accepted = sum(item.accepted for item in prediction.grasps) if prediction else 0
    if accepted == 0:
        cv2.putText(display, "REJECT: target not found or no safe grasp",
                    (12, display.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2, cv2.LINE_AA)
    return display


def save_result(output: Path, image: np.ndarray, prompt: str, objects, prediction,
                extra: dict | None = None):
    output.mkdir(parents=True, exist_ok=True)
    overlay = render(image, objects, prediction)
    if not cv2.imwrite(str(output / "grasp_overlay.jpg"), overlay):
        raise OSError("failed to write grasp_overlay.jpg")
    object_json = []
    for index, obj in enumerate(objects, 1):
        mask_name = f"mask_{index:03d}.png"
        cv2.imwrite(str(output / mask_name), obj.mask.astype(np.uint8) * 255)
        object_json.append({
            "id": index, "box": list(obj.box), "clipseg_score": obj.clipseg_score,
            "sam_score": obj.sam_score, "mask": mask_name,
            "target_class": obj.target_class, "target_score": obj.target_score,
            "predicted_class": obj.predicted_class,
            "competing_class": obj.competing_class,
            "competing_score": obj.competing_score,
            "semantic_margin": obj.semantic_margin,
            "semantic_accepted": obj.semantic_accepted,
            "semantic_backend": obj.semantic_backend,
            "localization_prompt": obj.localization_prompt,
            "classification_prompts": obj.classification_prompts,
            "similarities": obj.class_scores,
            "grasp": prediction.grasps[index - 1].as_dict(),
        })
    if prediction:
        np.savez_compressed(output / "maps.npz", **prediction.maps)
        for name in ("quality", "angle", "width"):
            cv2.imwrite(str(output / f"{name}.png"), heatmap(name, prediction.maps[name]))
        union = np.logical_or.reduce([obj.mask for obj in objects])
        cv2.imwrite(str(output / "quality_masked.png"),
                    (np.clip(prediction.maps["quality"], 0, 1) * union * 255).astype(np.uint8))
    semantic_count = sum(item.semantic_accepted for item in objects)
    accepted_count = sum(item.accepted for item in prediction.grasps) if prediction else 0
    metadata = {
        "prompt": prompt, "object_count": len(objects),
        "semantic_accepted_count": semantic_count,
        "accepted_object_count": accepted_count,
        "objects": object_json,
        "transform": prediction.transform if prediction else None,
        "model": prediction.backend if prediction else None,
        "robot_motion_authorized": False,
        "warning": "visual research output only; no collision or robot validation",
    }
    if extra:
        metadata.update(extra)
    (output / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return overlay, metadata
