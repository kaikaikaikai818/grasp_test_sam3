from __future__ import annotations

from dataclasses import dataclass, field
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
    target_class: str = ""
    target_score: float = 0.0
    competing_class: str = ""
    competing_score: float = 0.0
    semantic_margin: float = 0.0
    semantic_accepted: bool = True
    rejection_reason: str | None = None
    class_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCategory:
    name: str
    prompt: str
    aliases: tuple[str, ...]
    graspable: bool = True


def _normalise_text(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def build_tool_catalog(raw_categories: dict | None) -> dict[str, ToolCategory]:
    catalog: dict[str, ToolCategory] = {}
    for name, raw in (raw_categories or {}).items():
        prompt = str(raw.get("prompt", name.replace("_", " "))).strip()
        aliases = tuple({_normalise_text(name), _normalise_text(prompt), *(
            _normalise_text(str(alias)) for alias in raw.get("aliases", [])
        )})
        catalog[name] = ToolCategory(name, prompt, aliases, bool(raw.get("graspable", True)))
    return catalog


def resolve_tool_request(text: str, catalog: dict[str, ToolCategory]) -> ToolCategory:
    """Resolve a user phrase to one configured, graspable tool category."""
    query = _normalise_text(text)
    exact = [(item, len(query)) for item in catalog.values() if query in item.aliases]
    contained = [
        (item, max(len(alias) for alias in item.aliases if alias and alias in query))
        for item in catalog.values() if any(alias and alias in query for alias in item.aliases)
    ]
    matches = [(item, length) for item, length in (exact or contained) if item.graspable]
    if not matches:
        supported = ", ".join(item.prompt for item in catalog.values() if item.graspable)
        raise ValueError(f"unsupported tool request: {text!r}; supported tools: {supported}")
    return max(matches, key=lambda pair: pair[1])[0]


def masked_top_score(probability: np.ndarray, mask: np.ndarray, top_fraction: float) -> float:
    values = np.asarray(probability, dtype=float)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return 0.0
    count = max(1, int(np.ceil(values.size * top_fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def classify_mask(
    score_maps: dict[str, np.ndarray], mask: np.ndarray, target_class: str,
    min_score: float, min_margin: float, top_fraction: float,
) -> tuple[dict[str, float], bool, str | None]:
    scores = {
        name: masked_top_score(probability, mask, top_fraction)
        for name, probability in score_maps.items()
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ordered[0]
    target_score = scores[target_class]
    competitor_score = max((score for name, score in ordered if name != target_class), default=0.0)
    if target_score < min_score:
        return scores, False, f"target confidence {target_score:.3f} below {min_score:.3f}"
    if winner != target_class:
        return scores, False, f"looks more like {winner} ({winner_score:.3f})"
    margin = target_score - competitor_score
    if margin < min_margin:
        return scores, False, f"semantic margin {margin:.3f} below {min_margin:.3f}"
    return scores, True, None


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
                 max_mask_fraction=0.70, overlap_merge=0.20,
                 categories=None, semantic_min_score=0.35,
                 semantic_min_margin=0.03, semantic_top_fraction=0.25):
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
        self.catalog = build_tool_catalog(categories)
        if len(self.catalog) < 2:
            raise ValueError("recognition.categories must define at least two comparison classes")
        self.semantic_min_score = float(semantic_min_score)
        self.semantic_min_margin = float(semantic_min_margin)
        self.semantic_top_fraction = float(semantic_top_fraction)
        if not 0.0 < self.semantic_top_fraction <= 1.0:
            raise ValueError("semantic_top_fraction must be in (0, 1]")
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

    def _score_maps(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        height, width = rgb.shape[:2]
        categories = list(self.catalog.values())
        image = Image.fromarray(rgb)
        inputs = self.processor(
            text=[item.prompt for item in categories],
            images=[image] * len(categories),
            padding="max_length", return_tensors="pt",
        ).to("cpu")
        with torch.inference_mode():
            logits = self.detector(**inputs).logits.unsqueeze(1)
            probabilities = torch.nn.functional.interpolate(
                torch.sigmoid(logits), (height, width), mode="bilinear", align_corners=False
            )[:, 0].cpu().numpy()
        return {item.name: probabilities[index] for index, item in enumerate(categories)}

    def predict(self, image_bgr: np.ndarray, text: str) -> list[SamObject]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        target = resolve_tool_request(text, self.catalog)
        score_maps = self._score_maps(rgb)
        probability = score_maps[target.name]
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
        objects = merge_overlapping_objects(objects, self.overlap_merge)
        classified = []
        for item in objects:
            scores, accepted, reason = classify_mask(
                score_maps, item.mask, target.name,
                self.semantic_min_score, self.semantic_min_margin,
                self.semantic_top_fraction,
            )
            competitors = [(name, score) for name, score in scores.items() if name != target.name]
            competing_class, competing_score = max(competitors, key=lambda pair: pair[1])
            classified.append(SamObject(
                item.mask, item.box, item.clipseg_score, item.sam_score,
                target.name, scores[target.name], competing_class, competing_score,
                scores[target.name] - competing_score, accepted, reason, scores,
            ))
        return classified
