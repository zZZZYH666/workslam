#!/usr/bin/env python3
"""Generate prefix-7254 comparison plots including the provisional APE fix run.

The historical comparison directory is intentionally left untouched.  This
script writes a new, four-method comparison under the 2026-09-05 result
directory and uses the same timestamp mapping, common-frame intersection and
SE(3) alignment protocol as ``plot_w01_prefix7254_comparison.py``.
"""

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
OUT = ROOT / "results/result/w01_provisional_improvement_repeat5_2026-09-05/prefix7254_comparison"
FRAME_LIMIT = 7253
START, END = 5000, 6500

PATHS = {
    "baseline": ROOT / "results/result/w01_pre_rebuild_full_evaluation_2026-09-01/baseline/CameraTrajectory_baseline.txt",
    "adaptive_fallback": ROOT / "results/result/w01_pre_rebuild_full_evaluation_2026-09-01/adaptive_fallback/CameraTrajectory_semantic.txt",
    "provisional": ROOT / "results/result/w01_full_validation_fixed_2026-09-04/provisional/CameraTrajectory_semantic.txt",
    "provisional_ape_fix": ROOT / "results/result/w01_provisional_improvement_repeat5_2026-09-05/w01_semantic_fallback_repeat5_improved_fixed2/run_01/CameraTrajectory_semantic.txt",
}
LABELS = {
    "baseline": "Baseline ORB-SLAM3",
    "adaptive_fallback": "Adaptive fallback",
    "provisional": "Historical provisional",
    "provisional_ape_fix": "Provisional APE fix",
}
COLORS = {
    "baseline": "#666666",
    "adaptive_fallback": "#d97706",
    "provisional": "#1976a3",
    "provisional_ape_fix": "#8e24aa",
}


def compute():
    timestamps = ev.load_timestamps(ROOT / "data/W01_13Hz/timestampSecNanoSec_W01.txt", FRAME_LIMIT + 1)
    gt_all = ev.load_groundtruth(ROOT / "data/W01_13Hz/GT_W01.txt", FRAME_LIMIT + 1)
    trajectories = {}
    raw_counts = {}
    for name, path in PATHS.items():
        trajectories[name], raw_counts[name] = ev.trajectory_by_frame(path, timestamps, FRAME_LIMIT)
    common = np.array(sorted(set.intersection(*(set(v) for v in trajectories.values()))), dtype=int)
    gt = gt_all[common]
    aligned, errors = {}, {}
    for name, trajectory in trajectories.items():
        estimated = np.array([trajectory[i] for i in common])
        transform = ev.rigid_alignment(estimated[:, :3, 3], gt[:, :3, 3])
        aligned[name] = transform[None, :, :] @ estimated
        errors[name] = ev.absolute_errors(aligned[name], gt)
    return timestamps, common, gt, aligned, errors, raw_counts


def save_trajectory(gt, aligned, common):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
    ax.plot(gt[:, 0, 3], gt[:, 2, 3], color="#16697a", lw=2.5, label="Ground truth")
    for name in PATHS:
        pose = aligned[name]
        ax.plot(pose[:, 0, 3], pose[:, 2, 3], color=COLORS[name], lw=1.15, label=LABELS[name])
    ax.scatter(gt[0, 0, 3], gt[0, 2, 3], c="black", s=42, marker="o", label="Start", zorder=5)
    ax.scatter(gt[-1, 0, 3], gt[-1, 2, 3], c="#b42318", s=60, marker="X", label=f"End (frame {common[-1]})", zorder=5)
    ax.set_title("FinnForest W01 prefix trajectory comparison (common frames in 0–7253)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_comparison_xz.png")
    plt.close(fig)


def save_ape(errors, common):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=180, sharex=True)
    for name in PATHS:
        trans, rot = errors[name]
        axes[0].plot(common, trans, color=COLORS[name], lw=0.75, label=LABELS[name])
        axes[1].plot(common, rot, color=COLORS[name], lw=0.75, label=LABELS[name])
    axes[0].axvspan(START, END, color="#f59e0b", alpha=0.08, label="5000–6500")
    axes[0].set_ylabel("Translation APE [m]")
    axes[1].set_ylabel("Rotation APE [deg]")
    axes[1].set_xlabel("Frame index")
    axes[0].set_title("Absolute pose error after common-frame SE(3) alignment")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ape_translation_rotation.png")
    plt.close(fig)


def save_distribution(errors):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    names = list(PATHS)
    axes[0].boxplot([errors[n][0] for n in names], labels=[LABELS[n] for n in names], showfliers=False)
    axes[1].boxplot([errors[n][1] for n in names], labels=[LABELS[n] for n in names], showfliers=False)
    axes[0].set_title("Translation APE distribution")
    axes[0].set_ylabel("error [m]")
    axes[1].set_title("Rotation APE distribution")
    axes[1].set_ylabel("error [deg]")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(OUT / "ape_distribution_boxplot.png")
    plt.close(fig)


def save_rmse_comparison(errors):
    names = list(PATHS)
    x = np.arange(len(names))
    width = 0.36
    translation = [float(np.sqrt(np.mean(errors[n][0] ** 2))) for n in names]
    rotation = [float(np.sqrt(np.mean(errors[n][1] ** 2))) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    bars_t = axes[0].bar(x, translation, width, color=[COLORS[n] for n in names])
    bars_r = axes[1].bar(x, rotation, width, color=[COLORS[n] for n in names])
    axes[0].set_title("Translation APE RMSE comparison")
    axes[0].set_ylabel("RMSE [m]")
    axes[1].set_title("Rotation APE RMSE comparison")
    axes[1].set_ylabel("RMSE [deg]")
    for ax, bars in ((axes[0], bars_t), (axes[1], bars_r)):
        ax.set_xticks(x, [LABELS[n] for n in names], rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.suptitle("FinnForest W01 prefix APE RMSE comparison")
    fig.tight_layout()
    fig.savefig(OUT / "ape_rmse_comparison.png")
    plt.close(fig)


def save_rpe(aligned, gt, common, timestamps):
    rows = []
    for name in PATHS:
        rows.extend(ev.fixed_delta_rpe(name, aligned[name], gt, common, timestamps))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    for name in PATHS:
        selected = [r for r in rows if r["method"] == name]
        x = [r["delta_frames"] for r in selected]
        axes[0].plot(x, [r["translation_m_rmse"] for r in selected], marker="o", color=COLORS[name], label=LABELS[name])
        axes[1].plot(x, [r["rotation_deg_rmse"] for r in selected], marker="o", color=COLORS[name], label=LABELS[name])
    axes[0].set_title("Fixed-delta translation RPE")
    axes[0].set_ylabel("RMSE [m]")
    axes[1].set_title("Fixed-delta rotation RPE")
    axes[1].set_ylabel("RMSE [deg]")
    for ax in axes:
        ax.set_xlabel("frame delta")
        ax.set_xticks([1, 10, 100])
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rpe_fixed_delta_comparison.png")
    plt.close(fig)
    with (OUT / "rpe_fixed_delta.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_interval_ape(errors, common):
    mask = (common >= START) & (common <= END)
    frames = common[mask]
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=180)
    for name in PATHS:
        ax.plot(frames, errors[name][0][mask], lw=1.0, color=COLORS[name], label=LABELS[name])
    ax.axvspan(5500, 6000, color="#f59e0b", alpha=0.10, label="historical growth interval")
    ax.set_title("Translation APE comparison, frames 5000–6500")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Translation APE [m]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "translation_ape_5000_6500_comparison.png")
    plt.close(fig)

    rows = []
    for frame, index in zip(frames, np.flatnonzero(mask)):
        row = {"frame": int(frame)}
        for name in PATHS:
            row[f"{name}_translation_ape_m"] = float(errors[name][0][index])
        rows.append(row)
    with (OUT / "ape_interval_5000_6500.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_backend_diagnostics():
    run = ROOT / "results/result/w01_provisional_improvement_repeat5_2026-09-05/w01_semantic_fallback_repeat5_improved_fixed2/run_01"
    path = run / "backend_interval_stats.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    frame = np.array([int(r["frame_id"]) for r in rows])
    map_points = np.array([int(r["local_map_points"]) for r in rows])
    inliers = np.array([int(r["tracking_inliers"]) for r in rows])
    active = np.array([int(r["active_provisional"]) for r in rows])
    loop_matches = np.array([int(r["loop_matches"]) for r in rows])
    map_id = np.array([int(r["map_id"]) for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), dpi=180, sharex=True)
    axes[0].plot(frame, inliers, color="#16697a", lw=0.8, label="tracking inliers")
    axes[0].plot(frame, map_points, color="#455a64", lw=0.8, label="local map visible")
    axes[0].set_ylabel("count")
    axes[0].set_title("Provisional APE-fix backend diagnostics")
    axes[1].plot(frame, active, color="#8e24aa", lw=0.8, label="active provisional")
    axes[1].plot(frame, loop_matches, color="#2e7d32", lw=0.8, label="cumulative loop matches")
    axes[1].set_ylabel("count")
    axes[2].step(frame, map_id, where="post", color="#b42318", lw=1.0, label="map id")
    axes[2].set_ylabel("map id")
    axes[2].set_xlabel("Frame index")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "improved_backend_diagnostics.png")
    plt.close(fig)


def write_report(common, errors, raw_counts):
    lines = [
        "# W01 前 7254 帧对比图：Provisional APE 修复版",
        "",
        f"比较方法：Baseline、Adaptive fallback、历史 Provisional、Provisional APE fix。共同有效帧为 {len(common)}，范围为 {common[0]}–{common[-1]}。",
        "所有轨迹使用相同 GT 和时间戳映射，并在共同帧上分别进行 SE(3) 对齐。由于 APE 修复版在约 7213 帧发生地图重建，其共同帧少于完整 7254 帧；因此该图用于同协议诊断，不代表完整序列通过验收。",
        "",
        "## 图表",
        "",
    ]
    for filename, title in [
        ("trajectory_comparison_xz.png", "轨迹 x-z 对比"),
        ("ape_translation_rotation.png", "平移/旋转 APE 曲线"),
        ("ape_distribution_boxplot.png", "APE 分布箱线图"),
        ("ape_rmse_comparison.png", "APE RMSE 柱状图"),
        ("rpe_fixed_delta_comparison.png", "固定帧间隔 RPE"),
        ("translation_ape_5000_6500_comparison.png", "5000–6500 帧 Translation APE"),
        ("improved_backend_diagnostics.png", "改进版后端诊断"),
    ]:
        lines.append(f"- [{title}]({filename})")
    lines += [
        "",
        "## 主要统计",
        "",
        "| 方法 | 轨迹行数 | 共同帧覆盖 | 平移 APE RMSE (m) | 平移 P95 (m) | 旋转 APE RMSE (deg) | 旋转 P95 (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in PATHS:
        trans, rot = errors[name]
        coverage = 100.0 * len(common) / (FRAME_LIMIT + 1)
        lines.append(
            f"| {LABELS[name]} | {raw_counts[name]} | {coverage:.2f}% | "
            f"{np.sqrt(np.mean(trans**2)):.4f} | {np.percentile(trans,95):.4f} | "
            f"{np.sqrt(np.mean(rot**2)):.4f} | {np.percentile(rot,95):.4f} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- 这是与历史目录相同的共同帧、SE(3) 对齐、APE/RPE 计算协议。",
        "- Provisional APE fix 使用单次 `improved_fixed2/run_01`，不是 repeat-5 聚合结果。",
        "- 图中 APE 修复版的后段缺帧来自地图重建后的轨迹不连续，不能把共同帧对齐后的数值解释为完整 9210 帧性能结论。",
        "- `improved_backend_diagnostics.png` 使用改进版的逐帧后端 CSV，辅助定位约 7213 帧的地图重建。",
    ]
    (OUT / "轨迹APE_RPE对比报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timestamps, common, gt, aligned, errors, raw_counts = compute()
    save_trajectory(gt, aligned, common)
    save_ape(errors, common)
    save_distribution(errors)
    save_rmse_comparison(errors)
    save_rpe(aligned, gt, common, timestamps)
    save_interval_ape(errors, common)
    save_backend_diagnostics()
    write_report(common, errors, raw_counts)
    print(f"saved plots to {OUT}; common_frames={len(common)}; last_common_frame={common[-1]}")


if __name__ == "__main__":
    main()
