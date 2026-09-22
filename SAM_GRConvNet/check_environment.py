from pathlib import Path
import sys

import cv2
import numpy
import torch

from app import load_config


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, found {sys.version.split()[0]}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install/update the NVIDIA driver and PyTorch build.")
    config = load_config(Path(__file__).with_name("config.yaml"))
    missing = [path for path in (
        config["sam"]["checkpoint"], config["sam"]["clipseg_model"],
        config["grasp"]["checkpoint"],
    ) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing configured model path(s):\n" + "\n".join(missing))
    gpu = torch.cuda.get_device_properties(0)
    expected_gpu = config.get("environment", {}).get("expected_gpu", "").strip()
    if expected_gpu and expected_gpu.lower() not in gpu.name.lower():
        raise RuntimeError(f"Expected GPU '{expected_gpu}', found '{gpu.name}'")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} / CUDA build {torch.version.cuda}")
    print(f"GPU: {gpu.name} / {gpu.total_memory / 1024**3:.1f} GiB")
    print(f"NumPy: {numpy.__version__} / OpenCV: {cv2.__version__}")
    print("Configured model paths: OK")


if __name__ == "__main__":
    main()
