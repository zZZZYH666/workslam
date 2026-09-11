#!/usr/bin/env python3
"""Align and plot a FinnForest trajectory against its frame-wise ground truth."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aligned-output", type=Path)
    parser.add_argument("--matched-groundtruth-output", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--estimate-label", required=True)
    parser.add_argument("--max-time-difference-ns", type=float, default=1.0e6)
    return parser.parse_args()


def nearest_indices(reference, query):
    positions = np.searchsorted(reference, query)
    right = np.clip(positions, 0, len(reference) - 1)
    left = np.clip(positions - 1, 0, len(reference) - 1)
    choose_left = np.abs(reference[left] - query) <= np.abs(reference[right] - query)
    return np.where(choose_left, left, right)


def rigid_alignment(source, target):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean) / len(source)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    return (rotation @ source.T).T + translation


def main():
    args = parse_args()
    estimate = np.atleast_2d(np.loadtxt(args.estimate))
    poses = np.atleast_2d(np.loadtxt(args.groundtruth))
    timestamp_parts = np.atleast_2d(np.loadtxt(args.timestamps))
    groundtruth_timestamps = timestamp_parts[:, 0] * 1.0e9 + timestamp_parts[:, 1]
    groundtruth_xyz = poses[:, [3, 7, 11]]

    indices = nearest_indices(groundtruth_timestamps, estimate[:, 0])
    time_error = np.abs(groundtruth_timestamps[indices] - estimate[:, 0])
    valid = time_error <= args.max_time_difference_ns
    if not np.any(valid):
        raise RuntimeError("No estimated poses match the ground-truth timestamps")

    valid_indices = indices[valid]
    _, first_occurrences = np.unique(valid_indices, return_index=True)
    first_occurrences.sort()
    matched_indices = valid_indices[first_occurrences]
    matched_estimate = estimate[valid, 1:4][first_occurrences]
    matched_groundtruth = groundtruth_xyz[matched_indices]
    aligned_estimate = rigid_alignment(matched_estimate, matched_groundtruth)
    coverage = 100.0 * len(aligned_estimate) / len(groundtruth_xyz)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.aligned_output:
        np.savetxt(args.aligned_output, aligned_estimate, fmt="%.18e")
    if args.matched_groundtruth_output:
        np.savetxt(args.matched_groundtruth_output, matched_groundtruth, fmt="%.18e")

    fig, ax = plt.subplots(figsize=(12, 8.4), dpi=150)
    ax.plot(groundtruth_xyz[:, 0], groundtruth_xyz[:, 2], color="#1f77b4",
            linewidth=2.4, label="FinnForest ground truth")
    ax.plot(aligned_estimate[:, 0], aligned_estimate[:, 2], color="#ff7f0e",
            linewidth=1.25, label=f"{args.estimate_label} (SE(3)-aligned, {coverage:.1f}% coverage)")
    ax.set_title(args.title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.35)
    ax.axis("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"matched_poses={len(aligned_estimate)}")
    print(f"coverage_percent={coverage:.3f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
