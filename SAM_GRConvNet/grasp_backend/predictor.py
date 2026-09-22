from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

from .fusion import FusionResult, select_masked_grasp
from .postprocessing import decode_maps
from .preprocessing import prepare_rgb


CHECKPOINT_SHA256 = "b4a83e0c4b0db85bcbd70966f32807d8968b11cd18bd81ac2fc4c97046f8f680"


@dataclass(frozen=True)
class PredictionResult:
    maps: dict[str, np.ndarray]
    grasps: list[FusionResult]
    transform: dict[str, int]


class GraspPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", input_size: int = 224):
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        self.input_size = int(input_size)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and PyTorch build")
        vendor = Path(__file__).resolve().parents[1] / "third_party_promptgd"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        import clip
        self._clip = clip
        self.model = self._load_model()

    def _load_model(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"grasp checkpoint not found: {self.checkpoint}")
        with self.checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError("grasp checkpoint SHA256 does not match the verified author release")
        # The verified legacy release contains a complete nn.Module. Loading a
        # pickle is intentionally restricted to the exact known hash above.
        model = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        from inference.models.grconvnet3_CLIP import GenerativeResnet_CLIP
        if not isinstance(model, GenerativeResnet_CLIP):
            raise TypeError("checkpoint is not the expected GenerativeResnet_CLIP model")
        return model.float().eval().to(self.device)

    def predict(self, image, text: str, masks) -> PredictionResult:
        prompt = text.strip()
        if not prompt:
            raise ValueError("text prompt must not be empty")
        tensor, transform = prepare_rgb(image, self.input_size, self.device)
        tokens = self._clip.tokenize(prompt.split(), truncate=True).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(tensor, tokens)
        maps = decode_maps(outputs, transform)
        normalized_masks = [np.asarray(mask, dtype=bool) for mask in masks]
        grasps = [select_masked_grasp(maps, mask) for mask in normalized_masks]
        return PredictionResult(maps, grasps, transform.as_dict())
