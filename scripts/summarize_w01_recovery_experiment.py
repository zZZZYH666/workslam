#!/usr/bin/env python3
"""Evaluate recovery-timeout experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate_w01_multimap.py"
GT = ROOT / "data/W01_13Hz/GT_W01.txt"
TIMESTAMPS = ROOT / "data/W01_13Hz/timestampSecNanoSec_W01.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--last-frame", type=int, required=True)
    args = parser.parse_args()
    root = args.experiment_root if args.experiment_root.is_absolute() else ROOT / args.experiment_root
    rows = []
    for group in ("control_3s", "recovery_8s"):
        for run_dir in sorted((root / "runs" / group).glob("run_*")):
            files = sorted(run_dir.glob("trajectory_map_*_semantic.csv"))
            evaluation = run_dir / "multimap_evaluation.json"
            if files:
                command = ["python3", str(EVALUATOR)]
                for path in files:
                    command += ["--trajectory-with-map", str(path)]
                command += ["--groundtruth", str(GT), "--timestamps", str(TIMESTAMPS), "--last-frame", str(args.last_frame), "--output", str(evaluation)]
                subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            metadata = json.loads((run_dir / "run_metadata.json").read_text())
            summary = json.loads(evaluation.read_text()) if evaluation.exists() else {}
            frame_stats = next(iter(run_dir.glob("frame_stats_semantic.csv")), None)
            times = []
            states = {"2": 0, "3": 0, "4": 0}
            if frame_stats:
                with frame_stats.open() as stream:
                    for record in csv.DictReader(stream):
                        times.append(float(record["track_time_ms"]))
                        if record["tracking_state"] in states:
                            states[record["tracking_state"]] += 1
            global_ape = summary.get("global_ape") or {}
            rows.append({
                "group": group,
                "run": run_dir.name,
                "returncode": metadata.get("returncode"),
                "protocol": summary.get("protocol"),
                "coverage_percent": summary.get("coverage_percent"),
                "map_switch_count": summary.get("map_switch_count"),
                "contiguous_segment_count": summary.get("contiguous_segment_count"),
                "global_translation_rmse_m": global_ape.get("translation_rmse_m"),
                "global_rotation_rmse_deg": global_ape.get("rotation_rmse_deg"),
                "track_mean_ms": statistics.mean(times) if times else None,
                "track_p95_ms": statistics.quantiles(times, n=20, method="inclusive")[18] if len(times) >= 20 else None,
                "ok_frames": states["2"],
                "recently_lost_frames": states["3"],
                "lost_frames": states["4"],
            })
    output = root / "summary.csv"
    fields = list(rows[0].keys()) if rows else ["group"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
