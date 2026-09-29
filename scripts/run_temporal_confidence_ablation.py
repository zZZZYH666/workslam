#!/usr/bin/env python3
"""Run the strict A-E temporal-confidence ablation in order.

The runner intentionally stops starting new runs after the configured UTC
deadline, while allowing the currently running process to finish normally.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/2026-09-29_w01_temporal_confidence"
MASKS = RESULTS / "inputs/masks"
DATASET = ROOT / "data/W01_13Hz"
TIMESTAMPS = DATASET / "timestampSecNanoSec_W01.txt"
VOCAB = ROOT / "third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
BINARY = ROOT / "third_party/ORB_SLAM3/Examples/Stereo/finnforest_stereo_semantic"

GROUPS = [
    ("A", "A_baseline_fixed", "configs/finnforest_w01_stereo_rectified_turn_online.yaml"),
    ("B", "B_adaptive_fallback", "configs/finnforest_w01_stereo_rectified_adaptive_fallback.yaml"),
    ("C", "C_temporal_window", "configs/finnforest_w01_stereo_rectified_temporal_window.yaml"),
    ("D", "D_temporal_confidence", "configs/finnforest_w01_stereo_rectified_temporal_only.yaml"),
    ("E", "E_confidence_local_ba", "configs/finnforest_w01_stereo_rectified_temporal_confidence.yaml"),
]
PHASES = [("smoke200", 200), ("prefix4200", 4200), ("prefix7254", 7254), ("full9210", 9210)]
# A deadline is only active when explicitly supplied. This lets a resumed run
# continue the requested chain without inheriting the previous stop window.
DEADLINE_TEXT = os.environ.get("TEMPORAL_ABLATION_DEADLINE", "2099-12-31T23:59:59+00:00")
DEADLINE = datetime.fromisoformat(DEADLINE_TEXT)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def validate(out: Path, expected: int) -> dict:
    required = [
        "frame_stats_semantic.csv", "semantic_lifecycle_events.csv",
        "backend_interval_stats.csv", "frame_times_semantic.csv",
        "trajectory_maps_semantic.csv", "run_summary.csv",
        "CameraTrajectory_semantic.txt", "KeyFrameTrajectory_semantic.txt",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    checks = {"missing_outputs": missing, "expected_frames": expected}
    stats_path = out / "frame_stats_semantic.csv"
    timing_path = out / "frame_times_semantic.csv"
    if not missing:
        with stats_path.open(newline="") as handle:
            stats = list(csv.DictReader(handle))
        with timing_path.open(newline="") as handle:
            timing = list(csv.DictReader(handle))
        checks["stats_rows"] = len(stats)
        checks["timing_rows"] = len(timing)
        checks["frame_count_ok"] = len(stats) == expected and len(timing) == expected
        alpha = [float(row["fallback_alpha"]) for row in stats if row.get("fallback_alpha", "")]
        checks["max_fallback_alpha"] = max(alpha) if alpha else None
        checks["fallback_alpha_ok"] = all(math.isfinite(v) and v <= 0.500001 for v in alpha)
        with (out / "semantic_lifecycle_events.csv").open(newline="") as handle:
            lifecycle = list(csv.DictReader(handle))
        confidence = [float(row["confidence"]) for row in lifecycle if row.get("confidence", "")]
        checks["lifecycle_rows"] = len(lifecycle)
        checks["confidence_range_ok"] = all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in confidence)
        with (out / "backend_interval_stats.csv").open(newline="") as handle:
            backend = list(csv.DictReader(handle))
        checks["backend_rows"] = len(backend)
        checks["atlas_maps"] = max((int(row["atlas_maps"]) for row in backend), default=0)
        checks["tracking_states"] = sorted({row.get("tracking_state", "") for row in stats})
        checks["numeric_checks_ok"] = checks["fallback_alpha_ok"] and checks["confidence_range_ok"]
    else:
        checks.update({"frame_count_ok": False, "numeric_checks_ok": False})
    return checks


def run_one(letter: str, group: str, config_rel: str, phase: str, count: int) -> dict:
    started = now_utc()
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{phase}"
    out = RESULTS / group / run_id
    out.mkdir(parents=True, exist_ok=False)
    config = ROOT / config_rel
    shutil.copy2(config, out / "config.yaml")
    metadata = {
        "schema_version": 1, "experiment": group, "letter": letter,
        "run_id": run_id, "status": "running", "config": config_rel,
        "dataset": str(DATASET.relative_to(ROOT)), "masks": str(MASKS.relative_to(ROOT)),
        "timestamps": str(TIMESTAMPS.relative_to(ROOT)), "start_index": 0,
        "max_frames": count, "phase": phase, "started_utc": started.isoformat(),
    }
    write_json(out / "metadata.json", metadata)
    log_path = out / "run.log"
    cmd = [str(BINARY), str(VOCAB), str(config), str(DATASET), str(MASKS), str(TIMESTAMPS), str(out), str(count)]
    print(f"[{letter} {phase}] start {run_id}", flush=True)
    status = "completed"
    failure_reason = None
    with log_path.open("w") as log:
        completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        status = "failed"
        failure_reason = f"runner_exit_{completed.returncode}"
    checks = validate(out, count) if status == "completed" else {}
    if status == "completed" and not checks.get("frame_count_ok", False):
        status = "failed"
        failure_reason = "output_frame_count_or_required_csv_check_failed"
    if status == "completed" and not checks.get("numeric_checks_ok", False):
        status = "failed"
        failure_reason = "fallback_alpha_or_confidence_range_check_failed"
    metadata.update({"status": status, "finished_utc": now_utc().isoformat(), "checks": checks})
    if failure_reason:
        metadata["failure_reason"] = failure_reason
    write_json(out / "metadata.json", metadata)
    print(f"[{letter} {phase}] {status}; checks={json.dumps(checks, sort_keys=True)}", flush=True)
    return {"letter": letter, "group": group, "phase": phase, "run_id": run_id, "status": status, "checks": checks, "out": str(out)}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS / "execution_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest["resumed_utc"] = now_utc().isoformat()
        manifest["deadline_utc"] = DEADLINE.isoformat()
    else:
        manifest = {"started_utc": now_utc().isoformat(), "deadline_utc": DEADLINE.isoformat(), "runs": []}
    completed = {(run.get("letter"), run.get("phase")) for run in manifest.get("runs", []) if run.get("status") == "completed"}
    for phase, count in PHASES:
        for letter, group, config in GROUPS:
            if (letter, phase) in completed:
                print(f"[{letter} {phase}] already completed; skip", flush=True)
                continue
            if now_utc() >= DEADLINE:
                manifest["stopped_reason"] = "deadline_reached_before_next_run"
                write_json(manifest_path, manifest)
                print("deadline reached; no new run started", flush=True)
                return 0
            result = run_one(letter, group, config, phase, count)
            manifest["runs"].append(result)
            write_json(manifest_path, manifest)
            if result["status"] != "completed":
                manifest["stopped_reason"] = "run_failed"
                write_json(manifest_path, manifest)
                return 1
    manifest["finished_utc"] = now_utc().isoformat()
    manifest["stopped_reason"] = "all_runs_completed"
    write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
