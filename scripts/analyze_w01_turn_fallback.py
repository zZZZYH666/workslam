#!/usr/bin/env python3
"""Compare semantic filtering and the original all-feature tracker in the turn window."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["index"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=7190)
    parser.add_argument("--end", type=int, default=7320)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = read_rows(args.baseline)
    semantic = read_rows(args.semantic)
    rows = []
    for frame in range(args.start, args.end + 1):
        b = baseline[frame]
        s = semantic[frame]
        rows.append({
            "frame": frame,
            "baseline_features": int(b["tracked_keypoints"]),
            "baseline_inliers": int(b["tracking_inliers"]),
            "baseline_state": int(b["tracking_state"]),
            "semantic_static_left": int(s["static_left_features"]),
            "semantic_static_right": int(s["static_right_features"]),
            "semantic_fallback_left": int(s["fallback_left_features"]),
            "semantic_fallback_right": int(s["fallback_right_features"]),
            "semantic_stereo_matches": int(s["stereo_matches"]),
            "semantic_inliers": int(s["tracking_inliers"]),
            "semantic_state": int(s["tracking_state"]),
        })
    with (args.output / "turn_window_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def summary(items: list[dict[str, int]], prefix: str) -> dict[str, float | int]:
        inliers = [row[f"{prefix}_inliers"] for row in items]
        states = [row[f"{prefix}_state"] for row in items]
        features_key = f"{prefix}_features"
        result: dict[str, float | int] = {
            "frames": len(items),
            "mean_inliers": statistics.mean(inliers),
            "min_inliers": min(inliers),
            "ok_frames": sum(state == 2 for state in states),
            "recently_lost_frames": sum(state == 3 for state in states),
            "lost_frames": sum(state == 4 for state in states),
        }
        if features_key in items[0]:
            result["mean_features"] = statistics.mean(row[features_key] for row in items)
        return result

    all_rows = rows
    critical_rows = [row for row in rows if 7213 <= row["frame"] <= 7250]
    result = {
        "window": {"start": args.start, "end": args.end},
        "critical_failure_window": {"start": 7213, "end": 7250},
        "baseline_all_features": summary(all_rows, "baseline"),
        "semantic_filtered": summary(all_rows, "semantic"),
        "baseline_all_features_critical": summary(critical_rows, "baseline"),
        "semantic_filtered_critical": summary(critical_rows, "semantic"),
        "interpretation": {
            "baseline_is_existing_run": True,
            "not_an_online_turn_detector": True,
            "meaning": "The baseline is an empirical upper-bound reference for temporarily disabling semantic filtering during this difficult motion segment.",
        },
    }
    (args.output / "turn_window_comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
