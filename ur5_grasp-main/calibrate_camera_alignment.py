#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build D455-to-D435i relative alignment from the newest labelled log."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from bsp.camera_bsp.camera_alignment import fit_planar_alignment


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
# 留空时自动选择 outputs/validation 中最新的 measurements_*.jsonl。
MEASUREMENT_FILE = ""
OUTPUT_FILE = SCRIPT_ROOT / "camera_alignment.json"
MIN_SAMPLES_PER_POSITION = 3
EXPECTED_LABELS = ("center", "left", "right", "top", "bottom")


def find_measurement_file():
    if MEASUREMENT_FILE.strip():
        path = Path(MEASUREMENT_FILE.strip()).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    candidates = list((PROJECT_ROOT / "outputs" / "validation").glob("measurements_*.jsonl"))
    if not candidates:
        raise FileNotFoundError("没有找到 measurements_*.jsonl 测量文件。")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def load_grouped_medians(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    grouped = {}
    for row in rows:
        label = row.get("position_label")
        d455 = row.get("d455", {}).get("base_xyz_m")
        d435i = row.get("d435i", {}).get("base_xyz_m")
        if label in EXPECTED_LABELS and d455 is not None and d435i is not None:
            grouped.setdefault(label, []).append((d455, d435i))

    missing = [label for label in EXPECTED_LABELS
               if len(grouped.get(label, [])) < MIN_SAMPLES_PER_POSITION]
    if missing:
        detail = ", ".join("%s=%d" % (label, len(grouped.get(label, [])))
                           for label in EXPECTED_LABELS)
        raise ValueError("每个位置至少需要%d条，当前：%s；不足：%s"
                         % (MIN_SAMPLES_PER_POSITION, detail, ", ".join(missing)))

    source, target = [], []
    for label in EXPECTED_LABELS:
        pairs = grouped[label]
        source.append(np.median([pair[0] for pair in pairs], axis=0))
        target.append(np.median([pair[1] for pair in pairs], axis=0))
    return np.asarray(source), np.asarray(target), grouped


def main():
    measurement_path = find_measurement_file()
    source, target, grouped = load_grouped_medians(measurement_path)
    alignment = fit_planar_alignment(source, target)
    alignment.update({
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_measurement_file": str(measurement_path),
        "labels": list(EXPECTED_LABELS),
        "samples_per_label": {label: len(grouped[label]) for label in EXPECTED_LABELS},
        "d455_label_medians_m": source.tolist(),
        "d435i_label_medians_m": target.tolist(),
    })
    OUTPUT_FILE.write_text(json.dumps(alignment, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    before = np.linalg.norm(source - target, axis=1) * 1000.0
    after = np.asarray(alignment["fit_residual_m"]) * 1000.0
    angle_deg = np.degrees(np.arctan2(
        alignment["rotation_xy"][1][0], alignment["rotation_xy"][0][0]))
    print("校正完成。")
    print("输入文件:", measurement_path)
    print("输出文件:", OUTPUT_FILE)
    print("XY旋转: %.3f°" % angle_deg)
    print("XY平移: [%.2f, %.2f] mm" % tuple(
        np.asarray(alignment["translation_xy_m"]) * 1000.0))
    print("Z偏移: %.2f mm" % (alignment["z_offset_m"] * 1000.0))
    print("校正前距离: 中位数 %.2f mm，最大 %.2f mm" %
          (float(np.median(before)), float(np.max(before))))
    print("拟合后残差: 中位数 %.2f mm，最大 %.2f mm" %
          (float(np.median(after)), float(np.max(after))))
    print("下一步请重新采集一轮新位置数据，验证独立误差。")


if __name__ == "__main__":
    main()
