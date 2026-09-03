#!/usr/bin/env python3
"""Compare two FinnForest estimates on their common unique frame timestamps."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    return parser.parse_args()


def nearest_indices(reference, query):
    positions = np.searchsorted(reference, query)
    right = np.clip(positions, 0, len(reference) - 1)
    left = np.clip(positions - 1, 0, len(reference) - 1)
    return np.where(
        np.abs(reference[left] - query) <= np.abs(reference[right] - query),
        left,
        right,
    )


def unique_frame_positions(trajectory, groundtruth_timestamps):
    frame_indices = nearest_indices(groundtruth_timestamps, trajectory[:, 0])
    result = {}
    for frame_index, row in zip(frame_indices, trajectory):
        result.setdefault(int(frame_index), row[1:4])
    return result


def rigid_alignment(source, target):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean) / len(source)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return (rotation @ source.T).T + target_mean - rotation @ source_mean


def metrics(aligned, groundtruth):
    errors = np.linalg.norm(aligned - groundtruth, axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "ate_mean_m": float(np.mean(errors)),
        "ate_median_m": float(np.median(errors)),
        "ate_max_m": float(np.max(errors)),
        "aligned_path_length_m": float(np.linalg.norm(np.diff(aligned, axis=0), axis=1).sum()),
    }


def main():
    args = parse_args()
    baseline = np.atleast_2d(np.loadtxt(args.baseline))
    candidate = np.atleast_2d(np.loadtxt(args.candidate))
    poses = np.atleast_2d(np.loadtxt(args.groundtruth))
    timestamp_parts = np.atleast_2d(np.loadtxt(args.timestamps))
    groundtruth_timestamps = timestamp_parts[:, 0] * 1.0e9 + timestamp_parts[:, 1]
    groundtruth_xyz = poses[:, [3, 7, 11]]

    baseline_by_frame = unique_frame_positions(baseline, groundtruth_timestamps)
    candidate_by_frame = unique_frame_positions(candidate, groundtruth_timestamps)
    common_frames = np.array(sorted(set(baseline_by_frame) & set(candidate_by_frame)))
    shared_groundtruth = groundtruth_xyz[common_frames]
    baseline_xyz = np.array([baseline_by_frame[index] for index in common_frames])
    candidate_xyz = np.array([candidate_by_frame[index] for index in common_frames])
    aligned_baseline = rigid_alignment(baseline_xyz, shared_groundtruth)
    aligned_candidate = rigid_alignment(candidate_xyz, shared_groundtruth)

    groundtruth_length = float(np.linalg.norm(np.diff(shared_groundtruth, axis=0), axis=1).sum())
    rows = []
    for method, aligned in (("baseline", aligned_baseline), ("adaptive_fallback", aligned_candidate)):
        row = {"method": method, "common_frames": len(common_frames), **metrics(aligned, shared_groundtruth)}
        row["groundtruth_path_length_m"] = groundtruth_length
        row["path_length_error_percent"] = 100.0 * (row["aligned_path_length_m"] / groundtruth_length - 1.0)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(12, 8.4), dpi=150)
    ax.plot(shared_groundtruth[:, 0], shared_groundtruth[:, 2], color="#1f77b4",
            linewidth=2.5, label=f"FinnForest ground truth ({len(common_frames)} common frames)")
    ax.plot(aligned_baseline[:, 0], aligned_baseline[:, 2], color="#7f7f7f",
            linewidth=1.15, label="ORB-SLAM3 baseline (SE(3)-aligned)")
    ax.plot(aligned_candidate[:, 0], aligned_candidate[:, 2], color="#ff7f0e",
            linewidth=1.35, label="Adaptive semantic fallback (SE(3)-aligned)")
    ax.set_title("FinnForest W01: baseline vs adaptive semantic fallback")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.35)
    ax.axis("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"common_frames={len(common_frames)}")
    print(f"output={args.output}")
    print(f"metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
