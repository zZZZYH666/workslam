#!/usr/bin/env python3
"""Run the isolated W01 full-sequence stability experiment.

This runner uses the current FinnForest binary interfaces and stores every run
under a new experiment directory. It intentionally does not overwrite old
results and records command, hashes, environment, and return code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/W01_13Hz"
TIMESTAMPS = DATASET / "timestampSecNanoSec_W01.txt"
VOCABULARY = ROOT / "third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
BASELINE_BINARY = ROOT / "third_party/ORB_SLAM3/Examples/Stereo/finnforest_stereo_baseline"
SEMANTIC_BINARY = ROOT / "third_party/ORB_SLAM3/Examples/Stereo/finnforest_stereo_semantic"
BASELINE_CONFIG = ROOT / "configs/finnforest_w01_stereo_rectified.yaml"
GEOMETRIC_CONFIG = ROOT / "configs/finnforest_w01_stereo_rectified_geometric_promotion.yaml"
FUSED_MASKS = ROOT / "results/2026-09-04_w01_semantic_projection_sgbm_full_masks/fused_masks"
SOURCE_ROOT = ROOT / "third_party/ORB_SLAM3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def source_manifest() -> dict[str, str]:
    paths = sorted(path for directory in (SOURCE_ROOT / "src", SOURCE_ROOT / "include")
                   for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def run_group(name: str, count: int, start_run: int, root: Path, cpu_list: str, dry_run: bool) -> list[dict]:
    if count <= 0:
        return []
    semantic = name == "geometric"
    binary = SEMANTIC_BINARY if semantic else BASELINE_BINARY
    config = GEOMETRIC_CONFIG if semantic else BASELINE_CONFIG
    rows = []
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        env[key] = "1"
    env["OPENBLAS_MAIN_FREE"] = "1"
    for index in range(start_run, start_run + count):
        run_dir = root / "runs" / name / f"run_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = ["taskset", "-c", cpu_list, str(binary), str(VOCABULARY), str(config), str(DATASET)]
        if semantic:
            command += [str(FUSED_MASKS), str(TIMESTAMPS), str(run_dir)]
        else:
            command += [str(TIMESTAMPS), str(run_dir)]
        command += ["9210"]
        metadata = {
            "experiment": "w01_geometric_stability",
            "group": name,
            "run": index,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": command,
            "command_text": command_text(command),
            "cwd": str(ROOT),
            "cpu_list": cpu_list,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "binary": str(binary),
            "binary_sha256": sha256(binary),
            "config": str(config),
            "config_sha256": sha256(config),
            "vocabulary_sha256": sha256(VOCABULARY),
            "timestamps_sha256": sha256(TIMESTAMPS),
            "dataset": str(DATASET),
            "fused_masks": str(FUSED_MASKS) if semantic else None,
            "dry_run": dry_run,
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.copy2(config, run_dir / "config_used.yaml")
        if dry_run:
            print(command_text(command))
            rows.append({"group": name, "run": index, "returncode": None, "run_dir": str(run_dir)})
            continue
        log_path = run_dir / "run.log"
        started = time.time()
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        finished = time.time()
        metadata.update({"returncode": result.returncode, "start_epoch": started, "end_epoch": finished, "wall_time_seconds": finished - started})
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append({"group": name, "run": index, "returncode": result.returncode, "run_dir": str(run_dir)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-runs", type=int, default=3)
    parser.add_argument("--geometric-runs", type=int, default=5)
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--cpu-list", default="0-7")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "experiment_manifest.json"
    (root / "source_manifest.json").write_text(json.dumps({
        "source_root": str(SOURCE_ROOT),
        "files": source_manifest(),
    }, indent=2) + "\n")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["dry_run"] = args.dry_run
    else:
        manifest = {
        "experiment": "w01_geometric_stability",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "full_frame_count": 9210,
        "baseline_runs": 0,
        "geometric_runs": 0,
        "cpu_list": args.cpu_list,
        "dry_run": args.dry_run,
        "dataset": str(DATASET),
        "dataset_files": {"timestamps": sha256(TIMESTAMPS), "groundtruth": sha256(DATASET / "GT_W01.txt")},
        "runs": [],
        }
    new_runs = []
    new_runs += run_group("baseline", args.baseline_runs, args.start_run, root, args.cpu_list, args.dry_run)
    new_runs += run_group("geometric", args.geometric_runs, args.start_run, root, args.cpu_list, args.dry_run)
    existing_keys = {(item.get("group"), item.get("run")) for item in manifest.get("runs", [])}
    manifest["runs"] = [item for item in manifest.get("runs", []) if (item.get("group"), item.get("run")) not in {(row["group"], row["run"]) for row in new_runs}]
    manifest["runs"] += new_runs
    manifest["baseline_runs"] = len([item for item in manifest["runs"] if item.get("group") == "baseline"])
    manifest["geometric_runs"] = len([item for item in manifest["runs"] if item.get("group") == "geometric"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
