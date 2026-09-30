#!/usr/bin/env python3
"""Run the first E parameter-screening stage with reproducible metadata."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/2026-09-30_w01_e_parameter_tuning/parameter_runs"
BASE_CONFIG = ROOT / "configs/finnforest_w01_stereo_rectified_temporal_confidence.yaml"
MASKS = ROOT / "results/2026-09-29_w01_temporal_confidence/inputs/masks"
DATASET = ROOT / "data/W01_13Hz"
TIMESTAMPS = DATASET / "timestampSecNanoSec_W01.txt"
VOCAB = ROOT / "third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
BINARY = ROOT / "third_party/ORB_SLAM3/Examples/Stereo/finnforest_stereo_semantic"


W1 = {
    "Semantic.ConfidenceStaticWeight": "0.15",
    "Semantic.ConfidenceMatchWeight": "0.45",
    "Semantic.ConfidenceGeometryWeight": "0.40",
    "Semantic.ConfidenceDynamicWeight": "0.20",
    "Semantic.ConfidenceLambda": "0.60",
}

VARIANTS = {
    "E-P01": {},
    "E-P02": {
        "Semantic.ConfidencePromoteThreshold": "0.60",
        "Semantic.ConfidenceRejectThreshold": "0.10",
        "Semantic.PromotionMinMatchRatio": "0.50",
        "Semantic.PromotionMinStaticRatio": "0.50",
        "Semantic.ProvisionalMaxAgeKeyFrames": "12",
    },
    "E-P03": {
        "Semantic.ConfidencePromoteThreshold": "0.60",
        "Semantic.ConfidenceRejectThreshold": "0.10",
        "Semantic.GeometricPromotionMinVisibleFrames": "6",
        "Semantic.GeometricPromotionMinKeyFrames": "2",
        "Semantic.GeometricPromotionMinMatchRatio": "0.60",
        "Semantic.GeometricPromotionMaxReprojectionError": "1.75",
        "Semantic.ProvisionalMaxAgeKeyFrames": "12",
    },
    "E-P04": {
        "Semantic.ConfidencePromoteThreshold": "0.70",
        "Semantic.ConfidenceRejectThreshold": "0.15",
        "Semantic.PromotionMinMatchRatio": "0.55",
        "Semantic.ProvisionalMaxAgeKeyFrames": "12",
    },
    "E-P05": {
        "Semantic.ConfidencePromoteThreshold": "0.60",
        "Semantic.ConfidenceRejectThreshold": "0.10",
        "Semantic.PromotionMinMatchRatio": "0.50",
        "Semantic.PromotionMinStaticRatio": "0.50",
        "Semantic.ProvisionalMaxAgeKeyFrames": "12",
        **W1,
    },
    "E-P06": {
        "Semantic.ConfidencePromoteThreshold": "0.60",
        "Semantic.ConfidenceRejectThreshold": "0.10",
        "Semantic.GeometricPromotionMinVisibleFrames": "6",
        "Semantic.GeometricPromotionMinKeyFrames": "2",
        "Semantic.GeometricPromotionMinMatchRatio": "0.60",
        "Semantic.GeometricPromotionMaxReprojectionError": "1.75",
        "Semantic.ProvisionalMaxAgeKeyFrames": "12",
        **W1,
    },
}


def apply_overrides(text: str, overrides: dict[str, str]) -> str:
    for key, value in overrides.items():
        pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
        text, count = pattern.subn(f"{key}: {value}", text)
        if count != 1:
            raise RuntimeError(f"expected one config key, found {count}: {key}")
    return text


def validate_output(out: Path, expected_frames: int) -> dict:
    required = [
        "config.yaml", "metadata.json", "frame_stats_semantic.csv",
        "semantic_lifecycle_events.csv", "backend_interval_stats.csv",
        "frame_times_semantic.csv", "trajectory_maps_semantic.csv",
        "run_summary.csv", "CameraTrajectory_semantic.txt",
        "KeyFrameTrajectory_semantic.txt",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    checks = {"missing_outputs": missing, "expected_frames": expected_frames}
    if missing:
        checks.update({"frame_count_ok": False, "numeric_checks_ok": False})
        return checks
    with (out / "frame_stats_semantic.csv").open(newline="") as stream:
        stats = list(csv.DictReader(stream))
    with (out / "frame_times_semantic.csv").open(newline="") as stream:
        timing = list(csv.DictReader(stream))
    with (out / "semantic_lifecycle_events.csv").open(newline="") as stream:
        lifecycle = list(csv.DictReader(stream))
    alpha = [float(row["fallback_alpha"]) for row in stats]
    confidence = [float(row["confidence"]) for row in lifecycle if row.get("confidence", "")]
    checks["stats_rows"] = len(stats)
    checks["timing_rows"] = len(timing)
    checks["lifecycle_rows"] = len(lifecycle)
    checks["frame_count_ok"] = len(stats) == expected_frames and len(timing) == expected_frames
    checks["max_fallback_alpha"] = max(alpha, default=0.0)
    checks["fallback_alpha_ok"] = all(0.0 <= value <= 0.500001 for value in alpha)
    checks["confidence_range_ok"] = all(0.0 <= value <= 1.0 for value in confidence)
    checks["tracking_states"] = sorted({row.get("tracking_state", "") for row in stats})
    checks["numeric_checks_ok"] = checks["fallback_alpha_ok"] and checks["confidence_range_ok"]
    return checks


def run_variant(experiment_id: str, overrides: dict[str, str], max_frames: int) -> dict:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    phase = "smoke200" if max_frames == 200 else f"prefix{max_frames}"
    out = RESULTS / experiment_id / f"{stamp}_{phase}"
    out.mkdir(parents=True, exist_ok=False)
    config_text = apply_overrides(BASE_CONFIG.read_text(), overrides)
    (out / "config.yaml").write_text(config_text)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    metadata = {
        "schema_version": 1,
        "experiment": experiment_id,
        "stage": "lifecycle_parameter_screening",
        "status": "running",
        "config_source": str(BASE_CONFIG.relative_to(ROOT)),
        "dataset": str(DATASET.relative_to(ROOT)),
        "masks": str(MASKS.relative_to(ROOT)),
        "timestamps": str(TIMESTAMPS.relative_to(ROOT)),
        "start_index": 0,
        "max_frames": max_frames,
        "started_utc": now.isoformat(),
        "source_commit": commit,
        "parameter_overrides": overrides,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    command = [
        str(BINARY), str(VOCAB), str(out / "config.yaml"), str(DATASET),
        str(MASKS), str(TIMESTAMPS), str(out), str(max_frames),
    ]
    print(f"[{experiment_id}] start {out.name}", flush=True)
    with (out / "run.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    checks = validate_output(out, max_frames) if completed.returncode == 0 else {}
    status = "completed" if completed.returncode == 0 and checks.get("frame_count_ok") and checks.get("numeric_checks_ok") else "failed"
    metadata.update({
        "status": status,
        "return_code": completed.returncode,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    })
    if status == "failed":
        metadata["failure_reason"] = "runner_or_output_validation_failed"
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[{experiment_id}] {status} checks={json.dumps(checks, sort_keys=True)}", flush=True)
    return {"experiment": experiment_id, "output": str(out), **metadata}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--only", nargs="*", choices=sorted(VARIANTS))
    args = parser.parse_args()
    selected = args.only or list(VARIANTS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "lifecycle_parameter_screening",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "max_frames": args.max_frames,
        "runs": [],
    }
    for experiment_id in selected:
        result = run_variant(experiment_id, VARIANTS[experiment_id], args.max_frames)
        manifest["runs"].append({
            "experiment": experiment_id,
            "output": result["output"],
            "status": result["status"],
            "checks": result.get("checks", {}),
        })
        (RESULTS / "stage_lifecycle_screening_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        if result["status"] != "completed":
            return 1
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (RESULTS / "stage_lifecycle_screening_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
