#!/usr/bin/env python3
"""Complete accuracy and performance evaluation for the W01 prefix A/B run."""

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
DISTANCE_SEGMENTS_M = (100, 200, 300, 400, 500, 600, 700, 800)
FRAME_DELTAS = (1, 10, 100)


def parse_args():
    parser = argparse.ArgumentParser()
    for method in ("baseline", "candidate"):
        parser.add_argument(f"--{method}-trajectory", type=Path, required=True)
        parser.add_argument(f"--{method}-stats", type=Path, required=True)
        parser.add_argument(f"--{method}-keyframes", type=Path, required=True)
        parser.add_argument(f"--{method}-run-summary", type=Path, required=True)
        parser.add_argument(f"--{method}-resources", type=Path, required=True)
        parser.add_argument(f"--{method}-log", type=Path, required=True)
    parser.add_argument("--preprocessing-summary", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--last-frame", type=int, default=7253)
    return parser.parse_args()


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_timestamps(path, frame_count):
    rows = np.atleast_2d(np.loadtxt(path))[:frame_count]
    return np.rint(rows[:, 0] * 1.0e9 + rows[:, 1]).astype(np.int64)


def quaternion_matrix(quaternion):
    x, y, z, w = quaternion
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def row_to_pose(row):
    pose = np.eye(4)
    pose[:3, :3] = quaternion_matrix(row[4:8])
    pose[:3, 3] = row[1:4]
    return pose


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
            result.setdefault(frame, row_to_pose(row))
    return result, len(trajectory)


def load_groundtruth(path, frame_count):
    rows = np.atleast_2d(np.loadtxt(path))[:frame_count]
    poses = np.repeat(np.eye(4)[None, :, :], len(rows), axis=0)
    poses[:, :3, :] = rows.reshape(-1, 3, 4)
    return poses


def rigid_alignment(source_xyz, target_xyz):
    source_mean = source_xyz.mean(axis=0)
    target_mean = target_xyz.mean(axis=0)
    covariance = (source_xyz - source_mean).T @ (target_xyz - target_mean) / len(source_xyz)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    alignment = np.eye(4)
    alignment[:3, :3] = rotation
    alignment[:3, 3] = translation
    return alignment


def rotation_angles_deg(rotations):
    traces = np.trace(rotations, axis1=1, axis2=2)
    cosines = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def error_stats(values, prefix):
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def absolute_errors(estimated, groundtruth):
    translation = np.linalg.norm(estimated[:, :3, 3] - groundtruth[:, :3, 3], axis=1)
    rotation_error = np.transpose(groundtruth[:, :3, :3], (0, 2, 1)) @ estimated[:, :3, :3]
    rotation = rotation_angles_deg(rotation_error)
    return translation, rotation


def relative_error(estimated_first, estimated_last, gt_first, gt_last):
    estimated_delta = np.linalg.inv(estimated_first) @ estimated_last
    gt_delta = np.linalg.inv(gt_first) @ gt_last
    error = np.linalg.inv(estimated_delta) @ gt_delta
    translation = float(np.linalg.norm(error[:3, 3]))
    rotation = float(rotation_angles_deg(error[None, :3, :3])[0])
    return translation, rotation


def fixed_delta_rpe(method, estimated, groundtruth, frames, timestamps):
    rows = []
    for delta in FRAME_DELTAS:
        translations = []
        rotations = []
        durations = []
        for first in range(len(frames) - delta):
            last = first + delta
            if frames[last] - frames[first] != delta:
                continue
            translation, rotation = relative_error(
                estimated[first], estimated[last], groundtruth[first], groundtruth[last]
            )
            translations.append(translation)
            rotations.append(rotation)
            durations.append((timestamps[frames[last]] - timestamps[frames[first]]) * 1.0e-9)
        row = {
            "method": method,
            "delta_frames": delta,
            "mean_delta_seconds": float(np.mean(durations)),
            "samples": len(translations),
        }
        row.update(error_stats(translations, "translation_m"))
        row.update(error_stats(rotations, "rotation_deg"))
        rows.append(row)
    return rows


def distance_segment_rpe(method, estimated, groundtruth):
    positions = groundtruth[:, :3, 3]
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))))
    rows = []
    for length in DISTANCE_SEGMENTS_M:
        translations = []
        rotations = []
        for first in range(0, len(groundtruth), 10):
            last = int(np.searchsorted(distance, distance[first] + length, side="right"))
            if last >= len(groundtruth):
                break
            translation, rotation = relative_error(
                estimated[first], estimated[last], groundtruth[first], groundtruth[last]
            )
            translations.append(translation)
            rotations.append(rotation)
        rows.append({
            "method": method,
            "segment_m": length,
            "samples": len(translations),
            "translation_rpe_percent": 100.0 * float(np.mean(translations)) / length,
            "rotation_rpe_deg_per_m": float(np.mean(rotations)) / length,
            "translation_error_rmse_m": float(np.sqrt(np.mean(np.asarray(translations) ** 2))),
            "rotation_error_rmse_deg": float(np.sqrt(np.mean(np.asarray(rotations) ** 2))),
        })
    return rows


def count_lines(path):
    with path.open() as stream:
        return sum(1 for line in stream if line.strip())


def count_occurrences(path, needle):
    with path.open(errors="replace") as stream:
        return sum(line.count(needle) for line in stream)


def tracking_performance(method, rows, keyframes_path, run_summary_path, resources_path, log_path):
    times = np.array([float(row["track_time_ms"]) for row in rows])
    inliers = np.array([int(row["tracking_inliers"]) for row in rows])
    states = np.array([int(row["tracking_state"]) for row in rows])
    summary = load_csv(run_summary_path)[0]
    resources = load_csv(resources_path)[0]
    budget_ms = 1000.0 / 13.0
    wall_seconds = float(resources["wall_time_seconds"])
    keyframes = int(summary["current_map_keyframes"])
    map_points = int(summary["current_map_points"])
    return {
        "method": method,
        "frames": len(rows),
        "mean_track_time_ms": float(np.mean(times)),
        "median_track_time_ms": float(np.median(times)),
        "p95_track_time_ms": float(np.percentile(times, 95)),
        "p99_track_time_ms": float(np.percentile(times, 99)),
        "max_track_time_ms": float(np.max(times)),
        "track_fps_from_mean": 1000.0 / float(np.mean(times)),
        "frames_over_13hz_budget": int(np.sum(times > budget_ms)),
        "frames_over_13hz_budget_percent": 100.0 * float(np.mean(times > budget_ms)),
        "frames_over_100ms": int(np.sum(times > 100.0)),
        "frames_over_100ms_percent": 100.0 * float(np.mean(times > 100.0)),
        "mean_tracking_inliers": float(np.mean(inliers)),
        "median_tracking_inliers": float(np.median(inliers)),
        "p05_tracking_inliers": float(np.percentile(inliers, 5)),
        "ok_frames": int(np.sum(states == 2)),
        "recently_lost_frames": int(np.sum(states == 3)),
        "lost_frames": int(np.sum(states == 4)),
        "keyframe_trajectory_rows": count_lines(keyframes_path),
        "keyframe_rate_percent": 100.0 * keyframes / len(rows),
        "atlas_maps": int(summary["atlas_maps"]),
        "current_map_keyframes": keyframes,
        "current_map_points": map_points,
        "map_points_per_keyframe": map_points / keyframes,
        "loop_detections": count_occurrences(log_path, "*Loop detected"),
        "process_wall_time_seconds": wall_seconds,
        "process_wall_time_per_frame_ms": 1000.0 * wall_seconds / len(rows),
        "process_throughput_fps": len(rows) / wall_seconds,
        "user_cpu_seconds": float(resources["user_cpu_seconds"]),
        "system_cpu_seconds": float(resources["system_cpu_seconds"]),
        "total_cpu_seconds": float(resources["cpu_time_seconds"]),
        "average_cpu_percent": float(resources["average_cpu_percent"]),
        "peak_rss_mb": float(resources["peak_rss_mb"]),
    }


def preprocessing_performance(path, last_frame):
    rows = [row for row in load_csv(path) if int(row["index"]) <= last_frame]
    yolo = np.array([float(row["yolo_time_ms"]) for row in rows])
    post = np.array([float(row["total_time_ms"]) for row in rows])
    sgbm = np.array([float(row["sgbm_time_ms"]) for row in rows])
    combined = yolo + post
    return {
        "frames": len(rows),
        "mean_yolo_time_ms": float(np.mean(yolo)),
        "median_yolo_time_ms": float(np.median(yolo)),
        "p95_yolo_time_ms": float(np.percentile(yolo, 95)),
        "p99_yolo_time_ms": float(np.percentile(yolo, 99)),
        "max_yolo_time_ms": float(np.max(yolo)),
        "mean_sgbm_projection_time_ms": float(np.mean(sgbm)),
        "median_sgbm_projection_time_ms": float(np.median(sgbm)),
        "p95_sgbm_projection_time_ms": float(np.percentile(sgbm, 95)),
        "p99_sgbm_projection_time_ms": float(np.percentile(sgbm, 99)),
        "max_sgbm_projection_time_ms": float(np.max(sgbm)),
        "mean_postprocess_and_write_time_ms": float(np.mean(post)),
        "median_postprocess_and_write_time_ms": float(np.median(post)),
        "p95_postprocess_and_write_time_ms": float(np.percentile(post, 95)),
        "p99_postprocess_and_write_time_ms": float(np.percentile(post, 99)),
        "max_postprocess_and_write_time_ms": float(np.max(post)),
        "mean_offline_preprocessing_time_ms": float(np.mean(combined)),
        "median_offline_preprocessing_time_ms": float(np.median(combined)),
        "p95_offline_preprocessing_time_ms": float(np.percentile(combined, 95)),
        "p99_offline_preprocessing_time_ms": float(np.percentile(combined, 99)),
        "max_offline_preprocessing_time_ms": float(np.max(combined)),
        "offline_preprocessing_fps_from_mean": 1000.0 / float(np.mean(combined)),
    }


def plot_trajectory(path, groundtruth, baseline, candidate, frame_count):
    fig, ax = plt.subplots(figsize=(12, 8.4), dpi=160)
    ax.plot(groundtruth[:, 0, 3], groundtruth[:, 2, 3], color="#16697a", linewidth=2.4,
            label=f"FinnForest ground truth ({frame_count} common frames)")
    ax.plot(baseline[:, 0, 3], baseline[:, 2, 3], color="#666666", linewidth=1.25,
            label="Original ORB-SLAM3 (SE(3)-aligned)")
    ax.plot(candidate[:, 0, 3], candidate[:, 2, 3], color="#d97706", linewidth=1.4,
            label="Semantic projection + adaptive fallback (SE(3)-aligned)")
    ax.scatter(groundtruth[0, 0, 3], groundtruth[0, 2, 3], color="#16697a", s=42, label="Start")
    ax.scatter(groundtruth[-1, 0, 3], groundtruth[-1, 2, 3], color="#b42318", marker="X", s=60,
               label="Frame 7253")
    ax.set_title("FinnForest W01 complete pre-rebuild trajectory comparison (frames 0-7253)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ape(path, frames, errors):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=160, sharex=True)
    for method, color, label in (
        ("original_orbslam3", "#666666", "Original ORB-SLAM3"),
        ("semantic_adaptive_fallback", "#d97706", "Semantic projection + adaptive fallback"),
    ):
        axes[0].plot(frames, errors[method][0], color=color, linewidth=1.0, label=label)
        axes[1].plot(frames, errors[method][1], color=color, linewidth=1.0, label=label)
    axes[0].set_ylabel("Translation APE [m]")
    axes[1].set_ylabel("Rotation APE [deg]")
    axes[1].set_xlabel("Frame index")
    axes[0].set_title("FinnForest W01 absolute pose error (SE(3)-aligned)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_distance_rpe(path, rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    for method, color, label in (
        ("original_orbslam3", "#666666", "Original ORB-SLAM3"),
        ("semantic_adaptive_fallback", "#d97706", "Semantic projection + adaptive fallback"),
    ):
        selected = [row for row in rows if row["method"] == method]
        distances = [row["segment_m"] for row in selected]
        axes[0].plot(distances, [row["translation_rpe_percent"] for row in selected], marker="o",
                     color=color, label=label)
        axes[1].plot(distances, [row["rotation_rpe_deg_per_m"] for row in selected], marker="o",
                     color=color, label=label)
    axes[0].set_ylabel("Translation drift [%]")
    axes[1].set_ylabel("Rotation drift [deg/m]")
    for ax in axes:
        ax.set_xlabel("Segment length [m]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("FinnForest W01 distance-segment relative pose error")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_performance(path, rows):
    labels = ["Original\nORB-SLAM3", "Semantic\n+ fallback"]
    colors = ["#666666", "#d97706"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=160)
    metrics = (
        ("p95_track_time_ms", "TrackStereo P95 [ms]"),
        ("peak_rss_mb", "Peak process RSS [MB]"),
        ("current_map_keyframes", "Current-map keyframes"),
        ("current_map_points", "Current-map points"),
    )
    for ax, (key, title) in zip(axes.flat, metrics):
        values = [row[key] for row in rows]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.1f" if any(isinstance(value, float) for value in values) else "%d")
    fig.suptitle("FinnForest W01 SLAM performance and map-size comparison")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    frame_count = args.last_frame + 1
    timestamps = load_timestamps(args.timestamps, frame_count)
    groundtruth_all = load_groundtruth(args.groundtruth, frame_count)

    method_paths = {
        "original_orbslam3": args.baseline_trajectory,
        "semantic_adaptive_fallback": args.candidate_trajectory,
    }
    trajectories = {}
    output_lines = {}
    for method, path in method_paths.items():
        trajectories[method], output_lines[method] = trajectory_by_frame(path, timestamps, args.last_frame)
    common_frames = np.array(sorted(set.intersection(*(set(value) for value in trajectories.values()))), dtype=int)
    if len(common_frames) < 3:
        raise RuntimeError("At least three common trajectory frames are required")
    groundtruth = groundtruth_all[common_frames]

    aligned = {}
    ape_errors = {}
    accuracy_rows = []
    for method in method_paths:
        estimated = np.array([trajectories[method][frame] for frame in common_frames])
        alignment = rigid_alignment(estimated[:, :3, 3], groundtruth[:, :3, 3])
        aligned[method] = alignment[None, :, :] @ estimated
        translation_errors, rotation_errors = absolute_errors(aligned[method], groundtruth)
        ape_errors[method] = (translation_errors, rotation_errors)
        estimated_length = float(np.linalg.norm(np.diff(aligned[method][:, :3, 3], axis=0), axis=1).sum())
        gt_length = float(np.linalg.norm(np.diff(groundtruth[:, :3, 3], axis=0), axis=1).sum())
        row = {
            "method": method,
            "input_frames": frame_count,
            "trajectory_output_lines": output_lines[method],
            "unique_trajectory_frames": len(trajectories[method]),
            "coverage_percent": 100.0 * len(trajectories[method]) / frame_count,
            "common_evaluation_frames": len(common_frames),
        }
        row.update(error_stats(translation_errors, "translation_ape_m"))
        row.update(error_stats(rotation_errors, "rotation_ape_deg"))
        row.update({
            "endpoint_translation_error_m": float(translation_errors[-1]),
            "endpoint_rotation_error_deg": float(rotation_errors[-1]),
            "aligned_path_length_m": estimated_length,
            "groundtruth_path_length_m": gt_length,
            "path_length_error_percent": 100.0 * (estimated_length / gt_length - 1.0),
        })
        accuracy_rows.append(row)

    fixed_rpe_rows = []
    distance_rpe_rows = []
    for method in method_paths:
        fixed_rpe_rows.extend(fixed_delta_rpe(
            method, aligned[method], groundtruth, common_frames, timestamps
        ))
        distance_rpe_rows.extend(distance_segment_rpe(method, aligned[method], groundtruth))

    baseline_stats = [row for row in load_csv(args.baseline_stats) if int(row["index"]) <= args.last_frame]
    candidate_stats = [row for row in load_csv(args.candidate_stats) if int(row["index"]) <= args.last_frame]
    if len(baseline_stats) != frame_count or len(candidate_stats) != frame_count:
        raise RuntimeError("Stats CSV files do not contain exactly the requested prefix")
    performance_rows = [
        tracking_performance(
            "original_orbslam3", baseline_stats, args.baseline_keyframes,
            args.baseline_run_summary, args.baseline_resources, args.baseline_log,
        ),
        tracking_performance(
            "semantic_adaptive_fallback", candidate_stats, args.candidate_keyframes,
            args.candidate_run_summary, args.candidate_resources, args.candidate_log,
        ),
    ]
    preprocessing = preprocessing_performance(args.preprocessing_summary, args.last_frame)
    preprocessing["semantic_slam_mean_track_time_ms"] = performance_rows[1]["mean_track_time_ms"]
    preprocessing["estimated_preprocess_plus_track_mean_ms"] = (
        preprocessing["mean_offline_preprocessing_time_ms"] + performance_rows[1]["mean_track_time_ms"]
    )
    preprocessing["estimated_preprocess_plus_track_fps"] = (
        1000.0 / preprocessing["estimated_preprocess_plus_track_mean_ms"]
    )
    preprocessing["semantic_slam_process_wall_time_per_frame_ms"] = (
        performance_rows[1]["process_wall_time_per_frame_ms"]
    )
    preprocessing["estimated_sequential_full_process_mean_ms"] = (
        preprocessing["mean_offline_preprocessing_time_ms"]
        + performance_rows[1]["process_wall_time_per_frame_ms"]
    )
    preprocessing["estimated_sequential_full_process_fps"] = (
        1000.0 / preprocessing["estimated_sequential_full_process_mean_ms"]
    )

    fallback = np.array([int(row["fallback_used"]) for row in candidate_stats])
    fallback_rows = [{
        "frames": frame_count,
        "fallback_frames": int(np.sum(fallback)),
        "fallback_percent": 100.0 * float(np.mean(fallback)),
        "mean_static_left_features": float(np.mean([int(row["static_left_features"]) for row in candidate_stats])),
        "mean_static_right_features": float(np.mean([int(row["static_right_features"]) for row in candidate_stats])),
        "mean_fallback_left_features": float(np.mean([int(row["fallback_left_features"]) for row in candidate_stats])),
        "mean_fallback_right_features": float(np.mean([int(row["fallback_right_features"]) for row in candidate_stats])),
        "mean_stereo_matches": float(np.mean([int(row["stereo_matches"]) for row in candidate_stats])),
    }]

    state_rows = []
    for method, stats in (("original_orbslam3", baseline_stats),
                          ("semantic_adaptive_fallback", candidate_stats)):
        states = np.array([int(row["tracking_state"]) for row in stats])
        for state in sorted(set(states)):
            count = int(np.sum(states == state))
            state_rows.append({
                "method": method,
                "state_id": state,
                "state": STATE_NAMES.get(state, "UNKNOWN"),
                "frames": count,
                "percent": 100.0 * count / frame_count,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "accuracy_summary.csv", accuracy_rows)
    write_rows(args.output_dir / "rpe_fixed_delta.csv", fixed_rpe_rows)
    write_rows(args.output_dir / "rpe_distance_segments.csv", distance_rpe_rows)
    write_rows(args.output_dir / "performance_summary.csv", performance_rows)
    write_rows(args.output_dir / "semantic_preprocessing_performance.csv", [preprocessing])
    write_rows(args.output_dir / "fallback_summary.csv", fallback_rows)
    write_rows(args.output_dir / "tracking_state_counts.csv", state_rows)

    with (args.output_dir / "per_frame_ape.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "baseline_translation_m", "candidate_translation_m",
                         "baseline_rotation_deg", "candidate_rotation_deg"])
        writer.writerows(zip(
            common_frames,
            ape_errors["original_orbslam3"][0],
            ape_errors["semantic_adaptive_fallback"][0],
            ape_errors["original_orbslam3"][1],
            ape_errors["semantic_adaptive_fallback"][1],
        ))

    np.savetxt(args.output_dir / "groundtruth_common_poses.txt", groundtruth.reshape(len(groundtruth), 16), fmt="%.9f")
    for method in method_paths:
        np.savetxt(args.output_dir / f"aligned_{method}_poses.txt",
                   aligned[method].reshape(len(common_frames), 16), fmt="%.9f")

    plot_trajectory(
        args.output_dir / "trajectory_complete_comparison.png",
        groundtruth,
        aligned["original_orbslam3"],
        aligned["semantic_adaptive_fallback"],
        len(common_frames),
    )
    plot_ape(args.output_dir / "ape_translation_rotation.png", common_frames, ape_errors)
    plot_distance_rpe(args.output_dir / "rpe_distance_segments.png", distance_rpe_rows)
    plot_performance(args.output_dir / "slam_performance_comparison.png", performance_rows)

    print(f"common_frames={len(common_frames)}")
    for row in accuracy_rows:
        print(f"{row['method']}: translation_APE_RMSE={row['translation_ape_m_rmse']:.6f} m, "
              f"rotation_APE_RMSE={row['rotation_ape_deg_rmse']:.6f} deg")


if __name__ == "__main__":
    main()
