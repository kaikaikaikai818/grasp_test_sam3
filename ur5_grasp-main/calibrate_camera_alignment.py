#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create and independently validate one shared D455/D435i alignment file."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from bsp.camera_bsp.camera_alignment import (
    VALIDATION_MAX_LIMIT_M,
    VALIDATION_MEDIAN_LIMIT_M,
    alignment_metrics,
    build_calibration_context,
    load_alignment,
    select_candidate_alignment,
    validation_passes,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
OUTPUT_FILE = SCRIPT_ROOT / "camera_alignment.json"
MIN_SAMPLES_PER_POSITION = 3
EXPECTED_LABELS = ("center", "left", "right", "top", "bottom")
HO_SERIAL = "215122257404"
HI_SERIAL = "215222074676"
STREAM_SIZE = (640, 480)


def parse_args():
    parser = argparse.ArgumentParser(
        description="双相机一次性一致性验证；不会连接或移动机器人。")
    parser.add_argument("action", choices=("fit", "validate"),
                        help="fit生成候选；validate使用全新的采集记录独立验收")
    parser.add_argument("--measurement", type=Path,
                        help="指定 measurements_*.jsonl；默认自动选择合适的最新文件")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def measurement_candidates():
    folder = PROJECT_ROOT / "outputs" / "validation"
    return sorted(folder.glob("measurements_*.jsonl"),
                  key=lambda item: item.stat().st_mtime, reverse=True)


def resolve_measurement(requested, excluded=None, newer_than=None):
    if requested is not None:
        path = requested.expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError("测量文件不存在：%s" % path)
        if excluded is not None and path.resolve() == Path(excluded).resolve():
            raise ValueError("独立验收不能复用拟合数据，请重新采集五个位置。")
        if newer_than is not None and path.stat().st_mtime <= float(newer_than):
            raise ValueError("独立验收数据必须在候选补偿生成后重新采集。")
        return path
    for path in measurement_candidates():
        if (excluded is None or path.resolve() != Path(excluded).resolve()) and (
                newer_than is None or path.stat().st_mtime > float(newer_than)):
            return path
    if newer_than is not None:
        raise FileNotFoundError("没有找到候选补偿生成后采集的独立测量文件，请先执行菜单3。")
    raise FileNotFoundError("没有找到可用的 measurements_*.jsonl 测量文件。")


def load_grouped_medians(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    grouped = {}
    for row in rows:
        label = row.get("position_label")
        d455 = row.get("d455", {}).get("base_xyz_m")
        d435i = row.get("d435i", {}).get("base_xyz_m")
        serials_match = (
            row.get("d455", {}).get("serial") == HO_SERIAL
            and row.get("d435i", {}).get("serial") == HI_SERIAL)
        if (serials_match and label in EXPECTED_LABELS
                and d455 is not None and d435i is not None):
            grouped.setdefault(label, []).append((d455, d435i))

    missing = [label for label in EXPECTED_LABELS
               if len(grouped.get(label, [])) < MIN_SAMPLES_PER_POSITION]
    if missing:
        detail = ", ".join("%s=%d" % (label, len(grouped.get(label, [])))
                           for label in EXPECTED_LABELS)
        raise ValueError("每个位置至少需要%d条且序列号必须正确，当前：%s；不足：%s"
                         % (MIN_SAMPLES_PER_POSITION, detail, ", ".join(missing)))

    source, target = [], []
    for label in EXPECTED_LABELS:
        pairs = grouped[label]
        source.append(np.median([pair[0] for pair in pairs], axis=0))
        target.append(np.median([pair[1] for pair in pairs], axis=0))
    return np.asarray(source), np.asarray(target), grouped


def expected_context():
    return build_calibration_context(
        HO_SERIAL, HI_SERIAL, STREAM_SIZE, STREAM_SIZE,
        SCRIPT_ROOT / "camera_pose.txt",
        SCRIPT_ROOT / "cam2end_20260906.txt",
        SCRIPT_ROOT / "camera_20260906.ini",
    )


def print_metrics(title, metrics):
    values = np.asarray(metrics["residual_m"], dtype=float) * 1000.0
    print("%s：各位置=%s mm，中位数=%.2f mm，最大=%.2f mm" % (
        title, [round(value, 2) for value in values],
        metrics["median_m"] * 1000.0, metrics["max_m"] * 1000.0))


def fit_candidate(args):
    measurement = resolve_measurement(args.measurement)
    source, target, grouped = load_grouped_medians(measurement)
    alignment = select_candidate_alignment(source, target)
    fit_metrics = alignment_metrics(source, target, alignment)
    print_metrics("现有标定原始误差", alignment_metrics(source, target))
    print_metrics("候选补偿拟合残差", fit_metrics)
    fit_acceptable = (
        fit_metrics["max_m"] <= VALIDATION_MAX_LIMIT_M
        if alignment["mode"] == "validated_identity"
        else validation_passes(fit_metrics)
    )
    if not fit_acceptable:
        raise RuntimeError(
            "系统偏差拟合后仍无法达到中位数≤10mm且最大≤15mm；拒绝生成候选文件。"
            "请检查mask、深度、TCP和相机支架。")

    alignment.update({
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_measurement_file": str(measurement.resolve()),
        "labels": list(EXPECTED_LABELS),
        "samples_per_label": {label: len(grouped[label]) for label in EXPECTED_LABELS},
        "hardware_context": expected_context(),
        "validation": {"status": "pending", "message": "需要全新五位置数据独立验收"},
    })
    output = args.output.resolve()
    output.write_text(json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8")
    print("候选文件已生成：", output)
    print("当前仍禁止运动。请重新采集一轮五位置数据，再运行 validate。")


def validate_candidate(args):
    output = args.output.resolve()
    candidate = load_alignment(output, expected_context(), require_validated=False)
    if candidate is None:
        raise FileNotFoundError("候选文件不存在，请先运行 fit。")
    source_file = Path(candidate.get("source_measurement_file", ""))
    if not source_file.is_file():
        raise FileNotFoundError("候选文件记录的第一轮测量数据不存在，无法证明独立验收。")
    measurement = resolve_measurement(
        args.measurement,
        excluded=source_file,
        newer_than=source_file.stat().st_mtime,
    )
    source, target, grouped = load_grouped_medians(measurement)
    metrics = alignment_metrics(source, target, candidate)
    passed = validation_passes(metrics)
    candidate["validation"] = {
        "status": "passed" if passed else "failed",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "measurement_file": str(measurement.resolve()),
        "samples_per_label": {label: len(grouped[label]) for label in EXPECTED_LABELS},
        "residual_m": metrics["residual_m"],
        "median_m": metrics["median_m"],
        "max_m": metrics["max_m"],
        "limits_m": {
            "median": VALIDATION_MEDIAN_LIMIT_M,
            "maximum": VALIDATION_MAX_LIMIT_M,
        },
    }
    output.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    print_metrics("独立验收残差", metrics)
    if not passed:
        raise RuntimeError("独立验收未通过，文件保持锁定，禁止机器人运动。")
    print("独立验收通过。该文件可由所有工具共享：", output)


def main():
    args = parse_args()
    if args.action == "fit":
        fit_candidate(args)
    else:
        validate_candidate(args)


if __name__ == "__main__":
    main()
