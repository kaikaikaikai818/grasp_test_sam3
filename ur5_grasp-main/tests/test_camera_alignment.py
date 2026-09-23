import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsp.camera_bsp.camera_alignment import (
    ALIGNMENT_VERSION,
    alignment_metrics,
    apply_alignment_array,
    build_calibration_context,
    fit_planar_alignment,
    load_alignment,
    select_candidate_alignment,
    validation_passes,
)
from calibrate_camera_alignment import resolve_measurement


class CameraAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array([
            [0.40, -0.10, 0.02],
            [0.30, -0.10, 0.02],
            [0.50, -0.10, 0.02],
            [0.40, 0.00, 0.02],
            [0.40, -0.20, 0.02],
        ])

    def test_fit_recovers_planar_transform_and_z_offset(self):
        angle = np.deg2rad(4.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        target = apply_alignment_array(
            self.source, rotation, np.array([0.018, -0.012]), 0.006)
        alignment = fit_planar_alignment(self.source, target)
        fitted = apply_alignment_array(
            self.source,
            alignment["rotation_xy"],
            alignment["translation_xy_m"],
            alignment["z_offset_m"],
        )
        np.testing.assert_allclose(fitted, target, atol=1e-9)

    def test_small_raw_error_keeps_identity(self):
        target = self.source + np.array([0.004, -0.003, 0.002])
        alignment = select_candidate_alignment(self.source, target)
        self.assertEqual(alignment["mode"], "validated_identity")
        self.assertLess(alignment["raw_residual_max_m"], 0.015)

    def test_stable_large_offset_selects_fitted_correction(self):
        target = self.source + np.array([0.025, -0.018, 0.007])
        alignment = select_candidate_alignment(self.source, target)
        self.assertEqual(alignment["mode"], "fitted_correction")
        self.assertTrue(validation_passes(
            alignment_metrics(self.source, target, alignment)))

    def test_validation_requires_both_median_and_maximum(self):
        self.assertTrue(validation_passes({"median_m": 0.010, "max_m": 0.015}))
        self.assertFalse(validation_passes({"median_m": 0.011, "max_m": 0.012}))
        self.assertFalse(validation_passes({"median_m": 0.009, "max_m": 0.016}))

    def test_loader_rejects_pending_or_changed_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            core = [root / name for name in (
                "camera_pose.txt", "cam2end_20260906.txt", "camera_20260906.ini")]
            for index, path in enumerate(core):
                path.write_text("calibration-%d" % index, encoding="utf-8")
            context = build_calibration_context(
                "215122257404", "215222074676", (640, 480), (640, 480), *core)
            data = {
                "version": ALIGNMENT_VERSION,
                "rotation_xy": [[1.0, 0.0], [0.0, 1.0]],
                "translation_xy_m": [0.0, 0.0],
                "z_offset_m": 0.0,
                "hardware_context": context,
                "validation": {"status": "pending"},
            }
            output = root / "camera_alignment.json"
            output.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "尚未通过"):
                load_alignment(output, context)

            data["validation"] = {
                "status": "passed", "median_m": 0.006, "max_m": 0.012}
            output.write_text(json.dumps(data), encoding="utf-8")
            self.assertIsNotNone(load_alignment(output, context))

            core[0].write_text("changed", encoding="utf-8")
            changed_context = build_calibration_context(
                "215122257404", "215222074676", (640, 480), (640, 480), *core)
            with self.assertRaisesRegex(ValueError, "核心标定"):
                load_alignment(output, changed_context)

    def test_independent_measurement_must_be_newer_than_fit_data(self):
        with tempfile.TemporaryDirectory() as folder:
            old_path = Path(folder) / "measurements_old.jsonl"
            new_path = Path(folder) / "measurements_new.jsonl"
            old_path.write_text("{}\n", encoding="utf-8")
            cutoff = old_path.stat().st_mtime + 1.0
            with self.assertRaisesRegex(ValueError, "生成后重新采集"):
                resolve_measurement(old_path, newer_than=cutoff)
            new_path.write_text("{}\n", encoding="utf-8")
            future = cutoff + 1.0
            import os
            os.utime(new_path, (future, future))
            self.assertEqual(resolve_measurement(new_path, newer_than=cutoff), new_path)


if __name__ == "__main__":
    unittest.main()
