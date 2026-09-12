#!/usr/bin/env python3
"""Evaluate a geometric-promotion trajectory against the fixed W01 references."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evaluate_w01_full_accuracy_performance as ev


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "results/result/2026-09-01_w01_pre_rebuild_full_evaluation/baseline/CameraTrajectory_baseline.txt"
DEFAULT_ADAPTIVE = ROOT / "results/result/2026-09-01_w01_pre_rebuild_full_evaluation/adaptive_fallback/CameraTrajectory_semantic.txt"
INTERVALS = ((0, 7253), (5000, 5500), (5501, 6000), (6001, 6500), (6501, 6700), (5000, 6700), (7000, 7253))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", default="geometric_promotion")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--adaptive", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--groundtruth", type=Path, default=ROOT / "data/W01_13Hz/GT_W01.txt")
    parser.add_argument("--timestamps", type=Path, default=ROOT / "data/W01_13Hz/timestampSecNanoSec_W01.txt")
    parser.add_argument("--last-frame", type=int, default=7253)
    return parser.parse_args()


def stats(values, prefix):
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    methods = {
        "baseline": args.baseline,
        "adaptive_fallback": args.adaptive,
        args.candidate_name: args.candidate,
    }
    timestamps = ev.load_timestamps(args.timestamps, args.last_frame + 1)
    groundtruth_all = ev.load_groundtruth(args.groundtruth, args.last_frame + 1)
    trajectories = {
        name: ev.trajectory_by_frame(path, timestamps, args.last_frame)[0]
        for name, path in methods.items()
    }
    common = np.array(sorted(set.intersection(*(set(poses) for poses in trajectories.values()))), dtype=int)
    if len(common) < 3:
        raise RuntimeError("At least three common trajectory frames are required")

    groundtruth = groundtruth_all[common]
    errors = {}
    for name in methods:
        estimated = np.array([trajectories[name][frame] for frame in common])
        alignment = ev.rigid_alignment(estimated[:, :3, 3], groundtruth[:, :3, 3])
        aligned = alignment[None, :, :] @ estimated
        errors[name] = ev.absolute_errors(aligned, groundtruth)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy_rows = []
    for name, (translation, rotation) in errors.items():
        row = {"method": name, "common_frames": len(common), "first_frame": int(common[0]), "last_frame": int(common[-1])}
        row.update(stats(translation, "translation_ape_m"))
        row.update(stats(rotation, "rotation_ape_deg"))
        accuracy_rows.append(row)
    write_csv(args.output_dir / "accuracy_summary.csv", accuracy_rows)

    interval_rows = []
    for start, end in INTERVALS:
        if start > args.last_frame:
            continue
        selected = (common >= start) & (common <= min(end, args.last_frame))
        if not np.any(selected):
            continue
        for name, (translation, rotation) in errors.items():
            row = {"start_frame": start, "end_frame": min(end, args.last_frame), "method": name, "frames": int(np.sum(selected))}
            row.update(stats(translation[selected], "translation_ape_m"))
            row.update(stats(rotation[selected], "rotation_ape_deg"))
            interval_rows.append(row)
    write_csv(args.output_dir / "interval_ape_summary.csv", interval_rows)

    per_frame_rows = []
    for index, frame in enumerate(common):
        row = {"frame": int(frame)}
        for name, (translation, rotation) in errors.items():
            row[f"{name}_translation_ape_m"] = float(translation[index])
            row[f"{name}_rotation_ape_deg"] = float(rotation[index])
        per_frame_rows.append(row)
    write_csv(args.output_dir / "per_frame_ape.csv", per_frame_rows)
    np.savetxt(args.output_dir / "common_frames.txt", common, fmt="%d")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=160, sharex=True)
    colors = {"baseline": "#666666", "adaptive_fallback": "#d97706", args.candidate_name: "#16697a"}
    for name, (translation, rotation) in errors.items():
        axes[0].plot(common, translation, linewidth=0.9, color=colors[name], label=name)
        axes[1].plot(common, rotation, linewidth=0.9, color=colors[name], label=name)
    axes[0].set_ylabel("Translation APE [m]")
    axes[1].set_ylabel("Rotation APE [deg]")
    axes[1].set_xlabel("Frame")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "translation_rotation_ape.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
