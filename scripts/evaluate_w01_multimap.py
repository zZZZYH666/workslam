#!/usr/bin/env python3
"""Evaluate trajectories when a run may contain more than one map.

The evaluator deliberately separates global metrics from per-map metrics.  A
global APE/RPE is emitted only when every pose has a map id and a transform
from that map into one common frame is supplied.  Without those inputs the
tool reports local, independently aligned map segments and same-map RPE only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_timestamps(path: Path, frame_count: int) -> np.ndarray:
    rows = np.atleast_2d(np.loadtxt(path))[:frame_count]
    return np.rint(rows[:, 0] * 1.0e9 + rows[:, 1]).astype(np.int64)


def load_groundtruth(path: Path, frame_count: int) -> np.ndarray:
    rows = np.atleast_2d(np.loadtxt(path))[:frame_count]
    poses = np.repeat(np.eye(4)[None, :, :], len(rows), axis=0)
    poses[:, :3, :] = rows.reshape(-1, 3, 4)
    return poses


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_trajectory(path: Path, timestamps: np.ndarray, last_frame: int):
    rows = np.atleast_2d(np.loadtxt(path))
    query = np.rint(rows[:, 0]).astype(np.int64)
    right = np.clip(np.searchsorted(timestamps, query), 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    nearest = np.where(np.abs(timestamps[left] - query) <= np.abs(timestamps[right] - query), left, right)
    entries = []
    for frame, row in zip(nearest, rows):
        if int(frame) > last_frame:
            continue
        pose = np.eye(4)
        pose[:3, :3] = quaternion_matrix(row[4:8])
        pose[:3, 3] = row[1:4]
        entries.append((int(frame), pose, None))
    return entries


def read_trajectory_with_map(path: Path, last_frame: int):
    """Read the runner's explicit per-frame pose/map export."""
    entries = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            frame = int(row["frame"])
            if frame > last_frame or int(row.get("pose_valid", "1")) != 1:
                continue
            pose = np.eye(4)
            pose[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
            pose[:3, :3] = quaternion_matrix(np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]))
            entries.append((frame, pose, str(row["map_id"])))
    return entries


def read_trajectory_with_map_files(paths, last_frame: int):
    """Read and combine map-split runner exports.

    The caller may pass one file or one file per map. Frame IDs are deduplicated
    by the evaluation entry point after all files have been read.
    """
    entries = []
    for path in paths:
        entries.extend(read_trajectory_with_map(path, last_frame))
    return entries


def read_map_ids(path: Path | None):
    if path is None:
        return None
    mapping = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            mapping[int(row["frame"])] = str(row["map_id"])
    return mapping


def read_map_transforms(path: Path | None):
    if path is None:
        return None
    data = json.loads(path.read_text())
    result = {}
    for map_id, values in data.items():
        matrix = np.asarray(values, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"map {map_id} transform must be 4x4")
        result[str(map_id)] = matrix
    return result


def resolve_map_transforms(map_available, frames, map_ids, transforms):
    """Return transforms and provenance for the common estimated frame."""
    if not map_available or transforms is not None:
        return transforms, None
    unique_map_ids = sorted({map_ids[frame] for frame in frames})
    if len(unique_map_ids) == 1:
        return {unique_map_ids[0]: np.eye(4)}, "identity_single_map"
    return None, None


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean) / len(source)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_mean - rotation @ source_mean
    return transform


def sim3_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    xs = source - source_mean
    ys = target - target_mean
    covariance = xs.T @ ys / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    d = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        d[-1, -1] = -1
    rotation = vt.T @ d @ u.T
    variance = np.mean(np.sum(xs * xs, axis=1))
    scale = float(np.sum(singular * np.diag(d)) / variance) if variance > 0 else 1.0
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_mean - scale * rotation @ source_mean
    return transform


def rotation_error_deg(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    relative = np.transpose(truth[:, :3, :3], (0, 2, 1)) @ estimated[:, :3, :3]
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def normalize_pose_rotations(poses: np.ndarray) -> np.ndarray:
    """Remove Sim(3) scale from the rotational block before pose operations."""
    result = np.array(poses, dtype=float, copy=True)
    for i, pose in enumerate(result):
        u, _, vt = np.linalg.svd(pose[:3, :3])
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        result[i, :3, :3] = rotation
    return result


def summarize_ape(estimated: np.ndarray, truth: np.ndarray) -> dict:
    translation = np.linalg.norm(estimated[:, :3, 3] - truth[:, :3, 3], axis=1)
    rotation = rotation_error_deg(estimated, truth)
    return {
        "frames": int(len(translation)),
        "translation_rmse_m": float(np.sqrt(np.mean(translation ** 2))),
        "translation_median_m": float(np.median(translation)),
        "translation_p95_m": float(np.percentile(translation, 95)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation ** 2))),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p95_deg": float(np.percentile(rotation, 95)),
    }


def relative_error(estimated_first, estimated_last, truth_first, truth_last):
    estimated_delta = np.linalg.inv(estimated_first) @ estimated_last
    truth_delta = np.linalg.inv(truth_first) @ truth_last
    error = np.linalg.inv(estimated_delta) @ truth_delta
    translation = float(np.linalg.norm(error[:3, 3]))
    rotation = float(rotation_error_deg(error[None, ...], np.eye(4)[None, ...])[0])
    return translation, rotation


def same_map_rpe(frames, poses, truth, map_ids=None, deltas=(1, 10, 100)):
    index = {frame: i for i, frame in enumerate(frames)}
    rows = []
    for delta in deltas:
        translations, rotations = [], []
        for frame in frames:
            other = frame + delta
            if other not in index:
                continue
            if map_ids is not None and map_ids[frame] != map_ids[other]:
                continue
            t, r = relative_error(poses[index[frame]], poses[index[other]], truth[index[frame]], truth[index[other]])
            translations.append(t)
            rotations.append(r)
        rows.append({
            "delta_frames": delta,
            "scope": "same_map" if map_ids is not None else "unavailable_without_map_ids",
            "samples": len(translations),
            "translation_rmse_m": float(np.sqrt(np.mean(np.square(translations)))) if translations else None,
            "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rotations)))) if rotations else None,
        })
    return rows


def contiguous_segments(frames: list[int]):
    if not frames:
        return []
    segments = [[frames[0]]]
    for frame in frames[1:]:
        if frame == segments[-1][-1] + 1:
            segments[-1].append(frame)
        else:
            segments.append([frame])
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    trajectory_group = parser.add_mutually_exclusive_group(required=True)
    trajectory_group.add_argument("--trajectory", type=Path)
    trajectory_group.add_argument("--trajectory-with-map", type=Path, action="append",
                                  help="CSV exported by a FinnForest map; repeat for each map file")
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--map-ids", type=Path, help="CSV with frame,map_id")
    parser.add_argument("--map-transforms", type=Path, help="JSON map_id -> 4x4 global transform")
    parser.add_argument("--alignment", choices=("se3", "sim3"), default="se3")
    args = parser.parse_args()

    timestamps = load_timestamps(args.timestamps, args.last_frame + 1)
    truth_all = load_groundtruth(args.groundtruth, args.last_frame + 1)
    entries = (read_trajectory_with_map_files(args.trajectory_with_map, args.last_frame)
               if args.trajectory_with_map else
               read_trajectory(args.trajectory, timestamps, args.last_frame))
    # A repeated timestamp is an ambiguous export event, not another frame.
    by_frame = {}
    duplicate_rows = 0
    entry_map_ids = {}
    for frame, pose, map_id in entries:
        if frame in by_frame:
            duplicate_rows += 1
            continue
        by_frame[frame] = pose
        if map_id is not None:
            entry_map_ids[frame] = map_id
    frames = sorted(by_frame)
    map_ids = read_map_ids(args.map_ids)
    if map_ids is None and len(entry_map_ids) == len(frames):
        map_ids = entry_map_ids
    map_available = map_ids is not None and all(frame in map_ids for frame in frames)
    transforms = read_map_transforms(args.map_transforms)
    # A single map is already a common coordinate frame. Multi-map runs still
    # require an explicit transform for every map.
    transforms, global_transform_source = resolve_map_transforms(map_available, frames, map_ids, transforms)
    global_available = map_available and transforms is not None and all(map_ids[frame] in transforms for frame in frames)

    local_rows = []
    if map_available:
        groups = {}
        for frame in frames:
            groups.setdefault(map_ids[frame], []).append(frame)
    else:
        # Without map IDs, temporal contiguous segments are reported only as
        # export segments; they must not be called separate maps or combined
        # into one global trajectory.
        groups = {
            f"export_segment_{i + 1}": segment
            for i, segment in enumerate(contiguous_segments(frames))
        }
    for map_id, group_frames in groups.items():
        if len(group_frames) < 3:
            continue
        xyz = np.array([by_frame[f][:3, 3] for f in group_frames])
        gt = truth_all[group_frames]
        estimate = np.array([by_frame[f] for f in group_frames])
        transform = sim3_alignment(xyz, gt[:, :3, 3]) if args.alignment == "sim3" else rigid_alignment(xyz, gt[:, :3, 3])
        aligned = normalize_pose_rotations(transform[None, :, :] @ estimate)
        row = {"map_id": map_id, "first_frame": group_frames[0], "last_frame": group_frames[-1], "coverage_frames": len(group_frames)}
        row.update(summarize_ape(aligned, gt))
        local_rows.append(row)

    global_row = None
    rpe_rows = []
    switch_metrics = []
    if global_available:
        estimate = np.array([transforms[map_ids[f]] @ by_frame[f] for f in frames])
        gt = truth_all[frames]
        transform = sim3_alignment(estimate[:, :3, 3], gt[:, :3, 3]) if args.alignment == "sim3" else rigid_alignment(estimate[:, :3, 3], gt[:, :3, 3])
        aligned = normalize_pose_rotations(transform[None, :, :] @ estimate)
        global_row = summarize_ape(aligned, gt)
        rpe_rows = same_map_rpe(frames, aligned, gt, map_ids)
        for i in range(len(frames) - 1):
            if map_ids[frames[i]] == map_ids[frames[i + 1]]:
                continue
            switch_metrics.append({
                "from_map": map_ids[frames[i]],
                "to_map": map_ids[frames[i + 1]],
                "from_frame": frames[i],
                "to_frame": frames[i + 1],
                "frame_gap": frames[i + 1] - frames[i],
                "estimated_step_m": float(np.linalg.norm(aligned[i + 1, :3, 3] - aligned[i, :3, 3])),
                "groundtruth_step_m": float(np.linalg.norm(gt[i + 1, :3, 3] - gt[i, :3, 3])),
                "translation_jump_excess_m": float(np.linalg.norm(aligned[i + 1, :3, 3] - aligned[i, :3, 3]) - np.linalg.norm(gt[i + 1, :3, 3] - gt[i, :3, 3])),
            })
    elif map_available:
        estimate = np.array([by_frame[f] for f in frames])
        gt = truth_all[frames]
        rpe_rows = same_map_rpe(frames, estimate, gt, map_ids)

    segment_lengths = [len(segment) for segment in contiguous_segments(frames)]
    map_switch_count = None
    if map_available and frames:
        map_switch_count = sum(map_ids[a] != map_ids[b] for a, b in zip(frames, frames[1:]))
    result = {
        "protocol": "global" if global_available else "local_only",
        "alignment": args.alignment.upper(),
        "input_frames": args.last_frame + 1,
        "trajectory_rows": len(entries),
        "unique_frames": len(frames),
        "coverage_percent": 100.0 * len(frames) / (args.last_frame + 1),
        "duplicate_rows_discarded": duplicate_rows,
        "contiguous_segment_count": len(segment_lengths),
        "longest_contiguous_segment": max(segment_lengths, default=0),
        "longest_segment_percent": 100.0 * max(segment_lengths, default=0) / (args.last_frame + 1),
        "map_switch_count": map_switch_count,
        "map_id_available": map_available,
        "cross_map_transform_available": global_available,
        "global_transform_source": global_transform_source,
        "global_ape": global_row,
        "per_map_or_export_segment_ape": local_rows,
        "rpe": rpe_rows,
        "map_switch_events": switch_metrics,
        "warning": None if global_available else "Global APE/RPE is undefined; provide map IDs for every frame and complete cross-map transforms.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
