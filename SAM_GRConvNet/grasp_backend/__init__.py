"""Public API for the isolated GR-ConvNet/CLIP grasp backend."""

from .predictor import GraspPredictor, PredictionResult

__all__ = ["GraspPredictor", "PredictionResult"]
