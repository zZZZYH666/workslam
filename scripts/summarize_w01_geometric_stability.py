#!/usr/bin/env python3
"""Summarize W01 stability runs using the map-aware evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate_w01_multimap.py"
GT = ROOT / "data/W01_13Hz/GT_W01.txt"
TIMESTAMPS = ROOT / "data/W01_13Hz/timestampSecNanoSec_W01.txt"


def evaluate_run(run_dir: Path, group: str) -> dict:
    files = sorted(run_dir.glob(f"trajectory_map_*_{'semantic' if group == 'geometric' else 'baseline'}.csv"))
    output = run_dir / "multimap_evaluation.json"
    if not files:
        return {"group": group, "run": run_dir.name, "returncode": None, "evaluation_status": "missing_map_trajectory"}
    command = ["python3", str(EVALUATOR)]
    for path in files:
        command += ["--trajectory-with-map", str(path)]
    command += ["--groundtruth", str(GT), "--timestamps", str(TIMESTAMPS), "--last-frame", "9209", "--output", str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output.exists():
        return {"group": group, "run": run_dir.name, "returncode": result.returncode, "evaluation_status": "evaluator_failed", "evaluator_stderr": result.stderr[-1000:]}
    data = json.loads(output.read_text())
    summary = {
        "group": group,
        "run": run_dir.name,
        "returncode": json.loads((run_dir / "run_metadata.json").read_text()).get("returncode"),
        "protocol": data.get("protocol"),
        "coverage_percent": data.get("coverage_percent"),
        "unique_frames": data.get("unique_frames"),
        "map_id_available": data.get("map_id_available"),
        "cross_map_transform_available": data.get("cross_map_transform_available"),
        "map_switch_count": data.get("map_switch_count"),
        "contiguous_segment_count": data.get("contiguous_segment_count"),
        "longest_segment_percent": data.get("longest_segment_percent"),
        "global_translation_rmse_m": (data.get("global_ape") or {}).get("translation_rmse_m"),
        "global_rotation_rmse_deg": (data.get("global_ape") or {}).get("rotation_rmse_deg"),
        "local_segments": len(data.get("per_map_or_export_segment_ape", [])),
        "evaluation_status": "ok",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root if args.experiment_root.is_absolute() else ROOT / args.experiment_root
    rows = []
    for group in ("baseline", "geometric"):
        for run_dir in sorted((root / "runs" / group).glob("run_*")):
            rows.append(evaluate_run(run_dir, group))
    output = root / "summary.csv"
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    group_rows = []
    for group in ("baseline", "geometric"):
        selected = [row for row in rows if row.get("group") == group]
        def numeric(name):
            return [float(row[name]) for row in selected if row.get(name) not in (None, "")]
        coverages = numeric("coverage_percent")
        global_ape = numeric("global_translation_rmse_m")
        single_map = [row for row in selected if row.get("map_switch_count") == 0 and row.get("contiguous_segment_count") == 1 and row.get("coverage_percent") == 100.0]
        group_rows.append({
            "group": group,
            "runs": len(selected),
            "completed_runs": sum(row.get("returncode") == 0 for row in selected),
            "full_coverage_runs": sum(value == 100.0 for value in coverages),
            "single_map_success_runs": len(single_map),
            "single_map_success_rate": len(single_map) / len(selected) if selected else None,
            "map_switch_runs": sum((row.get("map_switch_count") or 0) > 0 for row in selected),
            "mean_coverage_percent": statistics.mean(coverages) if coverages else None,
            "mean_global_translation_rmse_m": statistics.mean(global_ape) if global_ape else None,
            "std_global_translation_rmse_m": statistics.stdev(global_ape) if len(global_ape) > 1 else None,
        })
    aggregate = root / "group_summary.csv"
    with aggregate.open("w", newline="") as stream:
        fields = list(group_rows[0].keys()) if group_rows else ["group"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(group_rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
