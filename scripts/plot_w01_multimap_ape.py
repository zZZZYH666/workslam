#!/usr/bin/env python3
"""Plot local (per export segment) APE for multi-map W01 runs.

Each segment is aligned independently with SE(3).  This is intentionally a
local-accuracy visualization; it must not be labelled as a global trajectory
APE unless map IDs and cross-map transforms are supplied separately.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_w01_multimap import (
    contiguous_segments,
    load_groundtruth,
    load_timestamps,
    read_trajectory,
    rigid_alignment,
    rotation_error_deg,
)


def parse_run(value: str):
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("run must be LABEL=TRAJECTORY_PATH")
    return label, Path(path)


def evaluate_run(label, path, timestamps, truth, last_frame):
    entries = read_trajectory(path, timestamps, last_frame)
    by_frame = {}
    duplicate_rows = 0
    for frame, pose, _map_id in entries:
        if frame in by_frame:
            duplicate_rows += 1
            continue
        by_frame[frame] = pose
    frames = sorted(by_frame)
    rows = []
    segments = contiguous_segments(frames)
    for segment_id, segment in enumerate(segments, 1):
        if len(segment) < 3:
            continue
        estimated = np.array([by_frame[frame] for frame in segment])
        groundtruth = truth[segment]
        transform = rigid_alignment(estimated[:, :3, 3], groundtruth[:, :3, 3])
        aligned = transform[None, :, :] @ estimated
        translation_errors = np.linalg.norm(aligned[:, :3, 3] - groundtruth[:, :3, 3], axis=1)
        rotation_errors = rotation_error_deg(aligned, groundtruth)
        for frame, translation_error, rotation_error in zip(segment, translation_errors, rotation_errors):
            rows.append({
                "label": label,
                "frame": frame,
                "translation_ape_m": float(translation_error),
                "rotation_ape_deg": float(rotation_error),
                "export_segment": segment_id,
            })
    values = np.array([row["translation_ape_m"] for row in rows], dtype=float)
    rotation_values = np.array([row["rotation_ape_deg"] for row in rows], dtype=float)
    summary = {
        "label": label,
        "trajectory_rows": len(entries),
        "unique_frames": len(frames),
        "coverage_percent": 100.0 * len(frames) / (last_frame + 1),
        "duplicate_rows_discarded": duplicate_rows,
        "segments": len(segments),
        "local_ape_frames": len(values),
        "local_translation_rmse_m": float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
        "local_translation_median_m": float(np.median(values)) if len(values) else None,
        "local_translation_p95_m": float(np.percentile(values, 95)) if len(values) else None,
        "local_rotation_rmse_deg": float(np.sqrt(np.mean(rotation_values ** 2))) if len(rotation_values) else None,
        "local_rotation_median_deg": float(np.median(rotation_values)) if len(rotation_values) else None,
        "local_rotation_p95_deg": float(np.percentile(rotation_values, 95)) if len(rotation_values) else None,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True,
                        help="LABEL=trajectory path; repeat for each run")
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    timestamps = load_timestamps(args.timestamps, args.last_frame + 1)
    truth = load_groundtruth(args.groundtruth, args.last_frame + 1)
    all_rows, summaries = [], []
    for label, path in args.run:
        rows, summary = evaluate_run(label, path, timestamps, truth, args.last_frame)
        all_rows.extend(rows)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "local_ape_per_frame.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "frame", "export_segment", "translation_ape_m", "rotation_ape_deg"])
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / "local_ape_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    colors = {
        "baseline": "#4d4d4d",
        "adaptive_run01": "#d97706",
        "adaptive_run02": "#f59e0b",
        "v2_run03": "#7c3aed",
        "geometric_run01": "#0f766e",
        "geometric_run02": "#0891b2",
        "geometric_run03": "#2563eb",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=180, sharex=True)
    for summary in summaries:
        label = summary["label"]
        points = [row for row in all_rows if row["label"] == label]
        if not points:
            continue
        grouped = {}
        for row in points:
            grouped.setdefault(row["export_segment"], []).append(row)
        for segment_id, segment_rows in grouped.items():
            frames = [r["frame"] for r in segment_rows]
            axes[0].plot(frames, [r["translation_ape_m"] for r in segment_rows],
                    lw=0.8, alpha=0.75, color=colors.get(label, None),
                    label=label if segment_id == min(grouped) else "_nolegend_")
            axes[1].plot(frames, [r["rotation_ape_deg"] for r in segment_rows],
                         lw=0.8, alpha=0.75, color=colors.get(label, None),
                         label=label if segment_id == min(grouped) else "_nolegend_")
    axes[0].set_ylabel("Translation APE [m]")
    axes[1].set_ylabel("Rotation APE [deg]")
    axes[1].set_xlabel("Frame")
    axes[0].set_title("W01 9/13 local APE (independent SE(3) alignment per export segment)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "local_translation_ape_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), dpi=180)
    labels = [row["label"] for row in summaries]
    values = [row["local_translation_rmse_m"] or np.nan for row in summaries]
    bars = axes[0].bar(labels, values, color=[colors.get(label, "#64748b") for label in labels])
    axes[0].set_ylabel("Translation APE RMSE [m]")
    values_rotation = [row["local_rotation_rmse_deg"] or np.nan for row in summaries]
    bars_rotation = axes[1].bar(labels, values_rotation, color=[colors.get(label, "#64748b") for label in labels])
    axes[1].set_ylabel("Rotation APE RMSE [deg]")
    axes[0].set_title("W01 9/13 local APE summary")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, value in zip(bars_rotation, values_rotation):
        if np.isfinite(value):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "local_translation_ape_rmse_summary.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
