#!/usr/bin/env python3
"""Evaluate baseline and semantic fallback runs on the W01 pre-rebuild prefix."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATE_NAMES = {
    0: "NO_IMAGES_YET",
    1: "NOT_INITIALIZED",
    2: "OK",
    3: "RECENTLY_LOST",
    4: "LOST",
    5: "OK_KLT",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--baseline-stats", type=Path, required=True)
    parser.add_argument("--baseline-keyframes", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-stats", type=Path, required=True)
    parser.add_argument("--candidate-keyframes", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--last-frame", type=int, default=7253)
    return parser.parse_args()


def load_timestamps(path):
    rows = np.atleast_2d(np.loadtxt(path))
    return np.rint(rows[:, 0] * 1.0e9 + rows[:, 1]).astype(np.int64)


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def trajectory_by_frame(path, timestamps, last_frame):
    trajectory = np.atleast_2d(np.loadtxt(path))
    query = np.rint(trajectory[:, 0]).astype(np.int64)
    positions = np.searchsorted(timestamps, query)
    right = np.clip(positions, 0, len(timestamps) - 1)
    left = np.clip(positions - 1, 0, len(timestamps) - 1)
    indices = np.where(
        np.abs(timestamps[left] - query) <= np.abs(timestamps[right] - query),
        left,
        right,
    )

    result = {}
    for frame, row in zip(indices, trajectory):
        frame = int(frame)
        if frame <= last_frame:
            result.setdefault(frame, row[1:4])
    return result, len(trajectory)


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


def percentile(values, value):
    return float(np.percentile(np.asarray(values, dtype=float), value))


def trajectory_metrics(aligned, groundtruth):
    errors = np.linalg.norm(aligned - groundtruth, axis=1)
    aligned_length = float(np.linalg.norm(np.diff(aligned, axis=0), axis=1).sum())
    groundtruth_length = float(np.linalg.norm(np.diff(groundtruth, axis=0), axis=1).sum())
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "ate_mean_m": float(np.mean(errors)),
        "ate_median_m": float(np.median(errors)),
        "ate_p95_m": percentile(errors, 95),
        "ate_max_m": float(np.max(errors)),
        "aligned_path_length_m": aligned_length,
        "groundtruth_path_length_m": groundtruth_length,
        "path_length_error_percent": 100.0 * (aligned_length / groundtruth_length - 1.0),
    }, errors


def runtime_metrics(rows):
    times = np.array([float(row["track_time_ms"]) for row in rows])
    return {
        "mean_track_time_ms": float(times.mean()),
        "median_track_time_ms": float(np.median(times)),
        "p95_track_time_ms": percentile(times, 95),
        "max_track_time_ms": float(times.max()),
        "mean_processing_fps": float(1000.0 / times.mean()),
    }


def tracking_metrics(rows):
    inliers = np.array([int(row["tracking_inliers"]) for row in rows])
    states = np.array([int(row["tracking_state"]) for row in rows])
    return {
        "mean_tracking_inliers": float(inliers.mean()),
        "median_tracking_inliers": float(np.median(inliers)),
        "p05_tracking_inliers": percentile(inliers, 5),
        "ok_frames": int(np.sum(states == 2)),
        "ok_percent": float(100.0 * np.mean(states == 2)),
        "recently_lost_frames": int(np.sum(states == 3)),
        "lost_frames": int(np.sum(states == 4)),
    }


def count_lines(path):
    with path.open() as stream:
        return sum(1 for line in stream if line.strip())


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectories(path, groundtruth, baseline, candidate, frame_count):
    fig, ax = plt.subplots(figsize=(12, 8.4), dpi=160)
    ax.plot(groundtruth[:, 0], groundtruth[:, 2], color="#16697a", linewidth=2.4,
            label=f"FinnForest ground truth ({frame_count} common frames)")
    ax.plot(baseline[:, 0], baseline[:, 2], color="#666666", linewidth=1.25,
            label="Original ORB-SLAM3 (SE(3)-aligned)")
    ax.plot(candidate[:, 0], candidate[:, 2], color="#d97706", linewidth=1.4,
            label="Semantic projection + adaptive fallback (SE(3)-aligned)")
    ax.scatter(groundtruth[0, 0], groundtruth[0, 2], color="#16697a", marker="o", s=45,
               zorder=5, label="Start")
    ax.scatter(groundtruth[-1, 0], groundtruth[-1, 2], color="#b42318", marker="X", s=60,
               zorder=5, label="Frame 7253")
    ax.set_title("FinnForest W01 pre-rebuild trajectory comparison (frames 0-7253)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_errors(path, frames, baseline_errors, candidate_errors):
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=160)
    ax.plot(frames, baseline_errors, color="#666666", linewidth=1.0, label="Original ORB-SLAM3")
    ax.plot(frames, candidate_errors, color="#d97706", linewidth=1.0,
            label="Semantic projection + adaptive fallback")
    ax.set_title("FinnForest W01 aligned absolute position error (frames 0-7253)")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Absolute position error [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    frame_count = args.last_frame + 1
    timestamps = load_timestamps(args.timestamps)[:frame_count]
    groundtruth_poses = np.atleast_2d(np.loadtxt(args.groundtruth))[:frame_count]
    groundtruth_xyz = groundtruth_poses[:, [3, 7, 11]]

    baseline_by_frame, baseline_lines = trajectory_by_frame(
        args.baseline_trajectory, timestamps, args.last_frame
    )
    candidate_by_frame, candidate_lines = trajectory_by_frame(
        args.candidate_trajectory, timestamps, args.last_frame
    )
    common_frames = np.array(sorted(set(baseline_by_frame) & set(candidate_by_frame)), dtype=int)
    if len(common_frames) < 3:
        raise RuntimeError("At least three common trajectory frames are required")

    shared_groundtruth = groundtruth_xyz[common_frames]
    baseline_xyz = np.array([baseline_by_frame[frame] for frame in common_frames])
    candidate_xyz = np.array([candidate_by_frame[frame] for frame in common_frames])
    aligned_baseline = rigid_alignment(baseline_xyz, shared_groundtruth)
    aligned_candidate = rigid_alignment(candidate_xyz, shared_groundtruth)
    baseline_trajectory_metrics, baseline_errors = trajectory_metrics(aligned_baseline, shared_groundtruth)
    candidate_trajectory_metrics, candidate_errors = trajectory_metrics(aligned_candidate, shared_groundtruth)

    baseline_stats = [row for row in load_csv(args.baseline_stats) if int(row["index"]) <= args.last_frame]
    candidate_stats = [row for row in load_csv(args.candidate_stats) if int(row["index"]) <= args.last_frame]
    if len(baseline_stats) != frame_count or len(candidate_stats) != frame_count:
        raise RuntimeError("Stats CSV files do not contain exactly the requested frame prefix")

    rows = []
    method_inputs = (
        ("original_orbslam3", baseline_by_frame, baseline_lines, args.baseline_keyframes,
         baseline_stats, baseline_trajectory_metrics),
        ("semantic_adaptive_fallback", candidate_by_frame, candidate_lines, args.candidate_keyframes,
         candidate_stats, candidate_trajectory_metrics),
    )
    for method, poses, output_lines, keyframes, stats, pose_metrics in method_inputs:
        row = {
            "method": method,
            "input_frames": frame_count,
            "trajectory_output_lines": output_lines,
            "unique_trajectory_frames": len(poses),
            "trajectory_coverage_percent": 100.0 * len(poses) / frame_count,
            "common_evaluation_frames": len(common_frames),
            "keyframes": count_lines(keyframes),
        }
        row.update(pose_metrics)
        row.update(tracking_metrics(stats))
        row.update(runtime_metrics(stats))
        rows.append(row)

    states = []
    for method, stats in (("original_orbslam3", baseline_stats),
                          ("semantic_adaptive_fallback", candidate_stats)):
        values = np.array([int(row["tracking_state"]) for row in stats])
        for state in sorted(set(values)):
            count = int(np.sum(values == state))
            states.append({
                "method": method,
                "state_id": state,
                "state": STATE_NAMES.get(state, "UNKNOWN"),
                "frames": count,
                "percent": 100.0 * count / frame_count,
            })

    fallback_counts = np.array([int(row["fallback_used"]) for row in candidate_stats])
    fallback_metrics = [{
        "input_frames": frame_count,
        "fallback_frames": int(fallback_counts.sum()),
        "fallback_percent": float(100.0 * fallback_counts.mean()),
        "mean_static_left_features": float(np.mean([int(row["static_left_features"]) for row in candidate_stats])),
        "mean_static_right_features": float(np.mean([int(row["static_right_features"]) for row in candidate_stats])),
        "mean_fallback_left_features": float(np.mean([int(row["fallback_left_features"]) for row in candidate_stats])),
        "mean_fallback_right_features": float(np.mean([int(row["fallback_right_features"]) for row in candidate_stats])),
        "mean_stereo_matches": float(np.mean([int(row["stereo_matches"]) for row in candidate_stats])),
    }]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "comparison_metrics.csv", rows)
    write_rows(args.output_dir / "tracking_state_counts.csv", states)
    write_rows(args.output_dir / "fallback_metrics.csv", fallback_metrics)
    np.savetxt(args.output_dir / "aligned_baseline_xyz.txt", aligned_baseline, fmt="%.9f")
    np.savetxt(args.output_dir / "aligned_adaptive_fallback_xyz.txt", aligned_candidate, fmt="%.9f")
    np.savetxt(args.output_dir / "groundtruth_common_xyz.txt", shared_groundtruth, fmt="%.9f")

    with (args.output_dir / "per_frame_ate.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "baseline_error_m", "adaptive_fallback_error_m"])
        writer.writerows(zip(common_frames, baseline_errors, candidate_errors))

    plot_trajectories(
        args.output_dir / "trajectory_baseline_vs_adaptive_fallback_pre_rebuild.png",
        shared_groundtruth,
        aligned_baseline,
        aligned_candidate,
        len(common_frames),
    )
    plot_errors(
        args.output_dir / "ate_by_frame_pre_rebuild.png",
        common_frames,
        baseline_errors,
        candidate_errors,
    )
    print(f"evaluated_frames={len(common_frames)}")
    for row in rows:
        print(f"{row['method']}: ATE_RMSE={row['ate_rmse_m']:.6f} m, "
              f"coverage={row['trajectory_coverage_percent']:.3f}%")


if __name__ == "__main__":
    main()
