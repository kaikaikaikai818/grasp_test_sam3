import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "summarize_grasp_trials", ROOT / "scripts" / "summarize_grasp_trials.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrialSummaryTests(unittest.TestCase):
    def test_aborted_instruction_counts_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            fields = ["trial_id", "tool", "correct_target", "grasped", "moved",
                      "placed", "stable_after_release"]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(20):
                    writer.writerow(dict(trial_id=index, tool="screwdriver",
                                         correct_target="yes" if index else "no",
                                         grasped="yes" if index else "",
                                         moved="yes" if index else "",
                                         placed="yes" if index else "",
                                         stable_after_release="yes" if index else ""))
            result = module.summarize(path)["screwdriver"]
            self.assertEqual(result["trials"], 20)
            self.assertEqual(result["successful"], 19)
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
