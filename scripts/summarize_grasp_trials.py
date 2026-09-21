"""Report per-tool first-attempt full-cycle success from an observed trial CSV."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

TOOLS = ("screwdriver", "adjustable wrench", "tape measure",
         "tape dispenser", "rubber mallet")


def summarize(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        if row.get("tool") not in TOOLS:
            raise ValueError("unknown tool: %s" % row.get("tool"))
        groups[row["tool"]].append(row)
    report = {}
    for tool in TOOLS:
        trials = groups[tool]
        successes = sum(all(row.get(field, "").strip().lower() == "yes"
                            for field in ("correct_target", "grasped", "moved",
                                          "placed", "stable_after_release"))
                        for row in trials)
        report[tool] = {"trials": len(trials), "successful": successes,
                        "rate": successes / len(trials) if trials else 0.0,
                        "passed": len(trials) >= 20 and successes / len(trials) >= .90}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    for tool, result in summarize(args.csv_file).items():
        print("%-19s %2d/%2d %5.1f%% %s" %
              (tool, result["successful"], result["trials"],
               result["rate"] * 100, "PASS" if result["passed"] else "PENDING"))


if __name__ == "__main__":
    main()
