#!/usr/bin/env python3
"""Run the single-variable recovery-timeout W01 experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/W01_13Hz"
TIMESTAMPS = DATASET / "timestampSecNanoSec_W01.txt"
VOCABULARY = ROOT / "third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
BINARY = ROOT / "third_party/ORB_SLAM3/Examples/Stereo/finnforest_stereo_semantic"
MASKS = ROOT / "results/2026-09-04_w01_semantic_projection_sgbm_full_masks/fused_masks"
CONFIGS = {
    "control_3s": ROOT / "configs/finnforest_w01_stereo_rectified_geometric_promotion.yaml",
    "recovery_8s": ROOT / "configs/finnforest_w01_stereo_rectified_geometric_recovery8.yaml",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4200)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--cpu-list", default="0-7")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        env[key] = "1"
    env["OPENBLAS_MAIN_FREE"] = "1"
    manifest = {
        "experiment": "w01_recovery_timeout",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frames": args.frames,
        "runs_per_group": args.runs,
        "cpu_list": args.cpu_list,
        "binary": str(BINARY),
        "binary_sha256": sha256(BINARY),
        "masks": str(MASKS),
        "timestamps_sha256": sha256(TIMESTAMPS),
        "groups": {},
    }
    for group, config in CONFIGS.items():
        manifest["groups"][group] = {"config": str(config), "config_sha256": sha256(config), "runs": []}
        for run_number in range(1, args.runs + 1):
            run_dir = root / "runs" / group / f"run_{run_number:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = ["taskset", "-c", args.cpu_list, str(BINARY), str(VOCABULARY), str(config), str(DATASET), str(MASKS), str(TIMESTAMPS), str(run_dir), str(args.frames)]
            metadata = {
                "group": group,
                "run": run_number,
                "frames": args.frames,
                "command": command,
                "cwd": str(ROOT),
                "binary_sha256": sha256(BINARY),
                "config_sha256": sha256(config),
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "platform": platform.platform(),
                "dry_run": args.dry_run,
            }
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            (run_dir / "config_used.yaml").write_text(config.read_text())
            if args.dry_run:
                print(" ".join(command))
                continue
            started = time.time()
            with (run_dir / "run.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            ended = time.time()
            metadata.update({"returncode": result.returncode, "start_epoch": started, "end_epoch": ended, "wall_time_seconds": ended - started})
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            manifest["groups"][group]["runs"].append({"run": run_number, "returncode": result.returncode, "run_dir": str(run_dir)})
    (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
