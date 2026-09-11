#!/usr/bin/env python3
"""Plot prefix-7254 trajectory/APE/RPE comparison for three W01 methods."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_w01_full_accuracy_performance as ev


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/result/w01_prefix7254_accuracy_comparison_2026-09-04"
FRAME_LIMIT = 7253

PATHS = {
    "baseline": ROOT / "results/result/w01_pre_rebuild_full_evaluation_2026-09-01/baseline/CameraTrajectory_baseline.txt",
    "adaptive_fallback": ROOT / "results/result/w01_pre_rebuild_full_evaluation_2026-09-01/adaptive_fallback/CameraTrajectory_semantic.txt",
    "provisional": ROOT / "results/result/w01_full_validation_fixed_2026-09-04/provisional/CameraTrajectory_semantic.txt",
}
LABELS = {
    "baseline": "Baseline ORB-SLAM3",
    "adaptive_fallback": "Adaptive fallback",
    "provisional": "Provisional MapPoint",
}
COLORS = {"baseline": "#666666", "adaptive_fallback": "#d97706", "provisional": "#1976a3"}


def compute():
    timestamps = ev.load_timestamps(ROOT / "data/W01_13Hz/timestampSecNanoSec_W01.txt", FRAME_LIMIT + 1)
    gt_all = ev.load_groundtruth(ROOT / "data/W01_13Hz/GT_W01.txt", FRAME_LIMIT + 1)
    trajectories = {name: ev.trajectory_by_frame(path, timestamps, FRAME_LIMIT)[0] for name, path in PATHS.items()}
    common = np.array(sorted(set.intersection(*(set(v) for v in trajectories.values()))), dtype=int)
    gt = gt_all[common]
    aligned, errors = {}, {}
    for name, trajectory in trajectories.items():
        estimated = np.array([trajectory[i] for i in common])
        transform = ev.rigid_alignment(estimated[:, :3, 3], gt[:, :3, 3])
        aligned[name] = transform[None, :, :] @ estimated
        errors[name] = ev.absolute_errors(aligned[name], gt)
    return timestamps, common, gt, aligned, errors


def save_trajectory(gt, aligned, common):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
    ax.plot(gt[:, 0, 3], gt[:, 2, 3], color="#16697a", lw=2.5, label="Ground truth")
    for name in PATHS:
        pose = aligned[name]
        ax.plot(pose[:, 0, 3], pose[:, 2, 3], color=COLORS[name], lw=1.25, label=LABELS[name])
    ax.scatter(gt[0, 0, 3], gt[0, 2, 3], c="black", s=42, marker="o", label="Start", zorder=5)
    ax.scatter(gt[-1, 0, 3], gt[-1, 2, 3], c="#b42318", s=60, marker="X", label=f"End (frame {common[-1]})", zorder=5)
    ax.set_title("FinnForest W01 prefix trajectory comparison (frames 0–7253)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.axis("equal"); ax.grid(alpha=.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(OUT / "trajectory_comparison_xz.png"); plt.close(fig)


def save_ape(errors, common):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=180, sharex=True)
    for name in PATHS:
        trans, rot = errors[name]
        axes[0].plot(common, trans, color=COLORS[name], lw=.8, label=LABELS[name])
        axes[1].plot(common, rot, color=COLORS[name], lw=.8, label=LABELS[name])
    axes[0].set_ylabel("Translation APE [m]"); axes[1].set_ylabel("Rotation APE [deg]"); axes[1].set_xlabel("Frame index")
    axes[0].set_title("Absolute pose error after SE(3) alignment")
    for ax in axes: ax.grid(alpha=.3); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "ape_translation_rotation.png"); plt.close(fig)


def save_distribution(errors):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=180)
    names = list(PATHS)
    axes[0].boxplot([errors[n][0] for n in names], labels=[LABELS[n] for n in names], showfliers=False)
    axes[1].boxplot([errors[n][1] for n in names], labels=[LABELS[n] for n in names], showfliers=False)
    axes[0].set_title("Translation APE distribution"); axes[0].set_ylabel("error [m]")
    axes[1].set_title("Rotation APE distribution"); axes[1].set_ylabel("error [deg]")
    for ax in axes: ax.grid(axis="y", alpha=.3); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(OUT / "ape_distribution_boxplot.png"); plt.close(fig)


def save_rmse_comparison(errors):
    names = list(PATHS)
    x = np.arange(len(names))
    width = 0.36
    translation = [float(np.sqrt(np.mean(errors[n][0] ** 2))) for n in names]
    rotation = [float(np.sqrt(np.mean(errors[n][1] ** 2))) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=180)
    bars_t = axes[0].bar(x, translation, width, color=[COLORS[n] for n in names])
    bars_r = axes[1].bar(x, rotation, width, color=[COLORS[n] for n in names])
    axes[0].set_title("Translation APE RMSE comparison")
    axes[0].set_ylabel("RMSE [m]")
    axes[1].set_title("Rotation APE RMSE comparison")
    axes[1].set_ylabel("RMSE [deg]")
    for ax, bars in ((axes[0], bars_t), (axes[1], bars_r)):
        ax.set_xticks(x, [LABELS[n] for n in names], rotation=15)
        ax.grid(axis="y", alpha=.3)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.suptitle("FinnForest W01 prefix APE RMSE comparison")
    fig.tight_layout(); fig.savefig(OUT / "ape_rmse_comparison.png"); plt.close(fig)


def save_rpe(aligned, gt, common, timestamps):
    rows = []
    for name in PATHS:
        rows.extend(ev.fixed_delta_rpe(name, aligned[name], gt, common, timestamps))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    for name in PATHS:
        selected = [r for r in rows if r["method"] == name]
        x = [r["delta_frames"] for r in selected]
        axes[0].plot(x, [r["translation_m_rmse"] for r in selected], marker="o", color=COLORS[name], label=LABELS[name])
        axes[1].plot(x, [r["rotation_deg_rmse"] for r in selected], marker="o", color=COLORS[name], label=LABELS[name])
    axes[0].set_title("Fixed-delta translation RPE"); axes[0].set_ylabel("RMSE [m]")
    axes[1].set_title("Fixed-delta rotation RPE"); axes[1].set_ylabel("RMSE [deg]")
    for ax in axes: ax.set_xlabel("frame delta"); ax.grid(alpha=.3); ax.legend(loc="best"); ax.set_xticks([1, 10, 100])
    fig.tight_layout(); fig.savefig(OUT / "rpe_fixed_delta_comparison.png"); plt.close(fig)
    with (OUT / "rpe_fixed_delta_plot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)


def write_report(common, errors):
    lines = ["# W01 前 7254 帧轨迹与定位误差图", "", f"三组共同有效帧：{len(common)}；范围：0–7253。所有轨迹先进行 SE(3) 对齐。", "", "## 图表", ""]
    for filename, title in [
        ("trajectory_comparison_xz.png", "轨迹 x-z 对比"),
        ("ape_translation_rotation.png", "平移/旋转 APE 曲线"),
        ("ape_distribution_boxplot.png", "APE 分布箱线图"),
        ("rpe_fixed_delta_comparison.png", "固定帧间隔 RPE"),
    ]:
        lines.append(f"- [{title}]({filename})")
    lines += ["", "## 主要统计", "", "| 方法 | 平移 APE RMSE (m) | 平移 P95 (m) | 旋转 APE RMSE (deg) | 旋转 P95 (deg) |", "|---|---:|---:|---:|---:|"]
    for name in PATHS:
        trans, rot = errors[name]
        lines.append(f"| {LABELS[name]} | {np.sqrt(np.mean(trans**2)):.4f} | {np.percentile(trans,95):.4f} | {np.sqrt(np.mean(rot**2)):.4f} | {np.percentile(rot,95):.4f} |")
    (OUT / "轨迹APE_RPE对比报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timestamps, common, gt, aligned, errors = compute()
    save_trajectory(gt, aligned, common)
    save_ape(errors, common)
    save_distribution(errors)
    save_rmse_comparison(errors)
    save_rpe(aligned, gt, common, timestamps)
    write_report(common, errors)
    print(f"saved plots to {OUT}; common_frames={len(common)}")


if __name__ == "__main__":
    main()
