"""Create a read-only size report for an LGD folder. This script deletes nothing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PATTERNS = ("lgd", "diffusion", "albef", "checkpoint", "best_model", "epoch_")


def category(path: Path) -> str:
    lowered = str(path).lower()
    if any(part in lowered for part in ("grasp-anything", "grasp_anything")):
        return "dataset_candidate"
    if any(part in lowered for part in ("\\exp\\", "\\runs\\", "\\logs\\", "\\wandb\\")):
        return "generated_output_candidate"
    if path.suffix.lower() in (".pt", ".pth", ".ckpt", ".bin") and any(
            token in lowered for token in PATTERNS):
        return "model_candidate"
    return "review_or_keep"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lgd_folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("lgd_inventory.json"))
    args = parser.parse_args()
    root = args.lgd_folder.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for path in root.rglob("*"):
        if path.is_file():
            rows.append({"path": str(path), "bytes": path.stat().st_size,
                         "category": category(path)})
    totals = {}
    for row in rows:
        totals[row["category"]] = totals.get(row["category"], 0) + row["bytes"]
    report = {
        "root": str(root), "deletes_files": False,
        "warning": "Candidates require manual path and usage review before deletion.",
        "totals_bytes": totals,
        "files": sorted(rows, key=lambda item: item["bytes"], reverse=True),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"只读清单已保存：{args.output.resolve()}")
    for name, size in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {size / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
