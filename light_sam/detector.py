from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    point: tuple[int, int]
    score: float
    area: float


class ClipSegDetector:
    """CPU-only text detector used as the replaceable V0 baseline."""

    def __init__(self, model_name: str = "CIDAS/clipseg-rd64-refined", threshold: float = 0.3,
                 min_area: float = 100.0, min_score: float = 0.35, max_objects: int = 10,
                 morphology_kernel: int = 5, local_files_only: bool = False) -> None:
        bundled_model = Path(__file__).resolve().parents[1] / "weights" / "clipseg-rd64-refined"
        self.model_name = str(bundled_model) if bundled_model.is_dir() else model_name
        self.threshold = threshold
        self.min_area = min_area
        self.min_score = min_score
        self.max_objects = max_objects
        self.morphology_kernel = morphology_kernel
        self.local_files_only = local_files_only
        self.device = torch.device("cpu")
        self.processor = CLIPSegProcessor.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        ).to(self.device)
        self.model.eval()

    def detect(self, image_rgb: np.ndarray, prompt: str) -> list[Detection]:
        pil_image = Image.fromarray(image_rgb)
        height, width = image_rgb.shape[:2]
        inputs = self.processor(
            text=[prompt], images=[pil_image], padding="max_length", return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits.unsqueeze(1))
            probabilities = torch.nn.functional.interpolate(
                probabilities, size=(height, width), mode="bilinear", align_corners=False
            )
        probability_map = probabilities[0, 0].cpu().numpy()

        binary_mask = (probability_map > self.threshold).astype(np.uint8) * 255
        kernel = np.ones((self.morphology_kernel, self.morphology_kernel), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(roi_mask, [shifted], -1, 255, thickness=cv2.FILLED)
            distance = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
            _, _, _, max_location = cv2.minMaxLoc(distance)
            point = (x + max_location[0], y + max_location[1])
            roi_probability = probability_map[y:y + h, x:x + w]
            score = float(roi_probability[roi_mask > 0].mean())
            if score < self.min_score:
                continue
            detections.append(Detection((x, y, x + w, y + h), point, score, area))
            if self.max_objects > 0 and len(detections) >= self.max_objects:
                break
        return detections
