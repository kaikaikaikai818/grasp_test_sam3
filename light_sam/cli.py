from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="轻量文字分割")
    parser.add_argument("--image", help="输入图片路径")
    parser.add_argument("--prompt", help="目标文字，建议使用英文")
    parser.add_argument("--output-dir", help="结果输出目录；默认按图片名自动创建")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min-area", type=float, default=100.0)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--morphology-kernel", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _clean_path(value: str) -> str:
    return value.strip().strip('"').strip("'")


def main() -> int:
    args = build_parser().parse_args()
    image = _clean_path(args.image or input("请输入图片路径: "))
    prompt = (args.prompt or input("请输入要识别的目标: ")).strip()
    if not image:
        raise ValueError("图片路径不能为空。")
    if not prompt:
        raise ValueError("目标文本不能为空。")

    output_dir = args.output_dir or str(Path("outputs") / Path(image).stem)
    metadata = run_pipeline(
        image_path=image,
        prompt=prompt,
        output_dir=output_dir,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        min_area=args.min_area,
        min_score=args.min_score,
        max_objects=args.max_objects,
        morphology_kernel=args.morphology_kernel,
        local_files_only=args.local_files_only,
    )
    print(f"处理完成，共找到 {metadata['object_count']} 个目标。")
    print(f"结果图片: {Path(output_dir) / 'sam_result.jpg'}")
    return 0
