from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from mobile_sam import SamPredictor, sam_model_registry


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def select_complete_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    point_xy: tuple[int, int],
    score_tolerance: float = 0.12,
    max_mask_fraction: float = 0.70,
) -> tuple[np.ndarray, float]:
    """Choose the largest credible SAM proposal containing the CLIPSeg seed.

    A tight CLIPSeg region often covers only the most distinctive part of a tool.
    Point-prompted SAM returns proposals at several semantic scales; selecting the
    largest high-scoring, non-background proposal recovers the complete object.
    """
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if masks.ndim != 3 or len(masks) == 0 or len(scores) != len(masks):
        raise ValueError("SAM masks and scores must have matching non-empty batches")

    height, width = masks.shape[1:]
    px = int(np.clip(point_xy[0], 0, width - 1))
    py = int(np.clip(point_xy[1], 0, height - 1))
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    fractions = areas / float(height * width)
    touches = np.stack(
        (masks[:, 0, :].any(1), masks[:, -1, :].any(1),
         masks[:, :, 0].any(1), masks[:, :, -1].any(1)), axis=1
    ).sum(axis=1)
    contains_seed = masks[:, py, px]

    valid = contains_seed & (areas > 0) & (fractions <= max_mask_fraction) & (touches < 2)
    if valid.any():
        best_valid_score = float(scores[valid].max())
        credible = valid & (scores >= best_valid_score - score_tolerance)
        index = int(np.flatnonzero(credible)[np.argmax(areas[credible])])
    else:
        index = int(np.argmax(scores))
    return masks[index], float(scores[index])


@dataclass(frozen=True)
class SamObject:
    mask: np.ndarray
    box: tuple[int, int, int, int]
    clipseg_score: float
    sam_score: float


def merge_overlapping_objects(
    objects: list[SamObject], overlap_threshold: float = 0.20
) -> list[SamObject]:
    """Merge duplicate SAM proposals for different parts of one object.

    The overlap coefficient uses the smaller mask as denominator, so a wrench
    head contained in a whole-handle proposal is treated as a duplicate while
    separate tools remain separate.
    """
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be between 0 and 1")
    merged: list[SamObject] = []
    for candidate in sorted(objects, key=lambda item: int(item.mask.sum()), reverse=True):
        duplicate_index = None
        for index, existing in enumerate(merged):
            intersection = int(np.logical_and(candidate.mask, existing.mask).sum())
            smaller = min(int(candidate.mask.sum()), int(existing.mask.sum()))
            if smaller and intersection / smaller >= overlap_threshold:
                duplicate_index = index
                break
        if duplicate_index is None:
            merged.append(candidate)
            continue
        existing = merged[duplicate_index]
        union = np.logical_or(existing.mask, candidate.mask)
        merged[duplicate_index] = SamObject(
            union,
            _mask_box(union),
            max(existing.clipseg_score, candidate.clipseg_score),
            max(existing.sam_score, candidate.sam_score),
        )
    return merged


class SamPipeline:
    def __init__(self, checkpoint, clipseg_model, device="cuda", threshold=0.3,
                 min_area=100, min_score=0.35, max_objects=10,
                 complete_object=True, sam_score_tolerance=0.12,
                 max_mask_fraction=0.70, overlap_merge=0.20):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MobileSAM requires CUDA")
        self.threshold = float(threshold)
        self.min_area = float(min_area)
        self.min_score = float(min_score)
        self.max_objects = int(max_objects)
        self.complete_object = bool(complete_object)
        self.sam_score_tolerance = float(sam_score_tolerance)
        self.max_mask_fraction = float(max_mask_fraction)
        self.overlap_merge = float(overlap_merge)
        model_path = Path(clipseg_model)
        if not model_path.is_dir():
            raise FileNotFoundError(f"CLIPSeg model directory not found: {model_path}")
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MobileSAM checkpoint not found: {checkpoint}")
        self.processor = CLIPSegProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.detector = CLIPSegForImageSegmentation.from_pretrained(
            str(model_path), local_files_only=True
        ).eval().to("cpu")
        sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
        self.predictor = SamPredictor(sam.eval().to(self.device))

    def predict(self, image_bgr: np.ndarray, text: str) -> list[SamObject]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        inputs = self.processor(
            text=[text], images=[Image.fromarray(rgb)], padding="max_length", return_tensors="pt"
        ).to("cpu")
        with torch.inference_mode():
            logits = self.detector(**inputs).logits.unsqueeze(1)
            probability = torch.nn.functional.interpolate(
                torch.sigmoid(logits), (height, width), mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()
        binary = (probability > self.threshold).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            local = np.zeros((h, w), np.uint8)
            cv2.drawContours(local, [contour - np.array([[[x, y]]])], -1, 255, cv2.FILLED)
            score = float(probability[y:y+h, x:x+w][local > 0].mean())
            if score < self.min_score:
                continue
            _, _, _, point = cv2.minMaxLoc(cv2.distanceTransform(local, cv2.DIST_L2, 5))
            detections.append(((x, y, x+w, y+h), (x+point[0], y+point[1]), score))
            if len(detections) >= self.max_objects:
                break
        if not detections:
            return []
        self.predictor.set_image(rgb)
        objects = []
        for box, point, clip_score in detections:
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([point]), point_labels=np.array([1]),
                    box=None if self.complete_object else np.array(box),
                    multimask_output=True,
                )
            if self.complete_object:
                mask, sam_score = select_complete_mask(
                    masks, scores, point,
                    self.sam_score_tolerance, self.max_mask_fraction,
                )
            else:
                best = int(np.argmax(scores))
                mask, sam_score = masks[best], float(scores[best])
            objects.append(SamObject(mask, _mask_box(mask), clip_score, sam_score))
        return merge_overlapping_objects(objects, self.overlap_merge)
