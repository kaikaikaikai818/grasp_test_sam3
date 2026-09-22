"""Run text-guided SAM segmentation and GR-ConvNet/CLIP on one image."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2

from app import build_models, infer, load_config, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/image"))
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"image not found: {args.image}")
    started = time.perf_counter()
    sam, grasp = build_models(load_config(args.config))
    objects, prediction = infer(image, args.text, sam, grasp)
    _, metadata = save_result(
        args.output, image, args.text, objects, prediction,
        {"image": str(args.image.resolve()), "seconds": time.perf_counter() - started},
    )
    print(f"完成：发现 {metadata['object_count']} 个目标。结果：{args.output.resolve()}")


if __name__ == "__main__":
    main()
