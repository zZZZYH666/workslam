#!/usr/bin/env python3
"""Analyze the provisional Translation APE expansion in frames 5000-6500."""
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
LAST = 7253
START, END = 5000, 6500


def main():
    timestamps = ev.load_timestamps(ROOT / "data/W01_13Hz/timestampSecNanoSec_W01.txt", LAST + 1)
    gt_all = ev.load_groundtruth(ROOT / "data/W01_13Hz/GT_W01.txt", LAST + 1)
    paths = {
        "baseline": ROOT / "results/result/w01_pre_rebuild_full_evaluation_2026-09-01/baseline/CameraTrajectory_baseline.txt",
        "adaptive_fallback": ROOT / "results/result/w01_pre_rebuild_full_evaluation_2026-09-01/adaptive_fallback/CameraTrajectory_semantic.txt",
        "provisional": ROOT / "results/result/w01_full_validation_fixed_2026-09-04/provisional/CameraTrajectory_semantic.txt",
    }
    trajectories = {name: ev.trajectory_by_frame(path, timestamps, LAST)[0] for name, path in paths.items()}
    common = np.array(sorted(set.intersection(*(set(v) for v in trajectories.values()))), dtype=int)
    gt = gt_all[common]
    errors = {}
    aligned = {}
    for name, trajectory in trajectories.items():
        estimated = np.array([trajectory[i] for i in common])
        transform = ev.rigid_alignment(estimated[:, :3, 3], gt[:, :3, 3])
        aligned[name] = transform[None, :, :] @ estimated
        errors[name] = ev.absolute_errors(aligned[name], gt)[0]

    stats_path = ROOT / "results/result/w01_full_validation_fixed_2026-09-04/provisional/frame_stats_semantic.csv"
    stats = {int(row["index"]): row for row in csv.DictReader(stats_path.open())}
    selected = [i for i, frame in enumerate(common) if START <= frame <= END]
    plot_frames = common[selected]

    rows = []
    for j in selected:
        frame = int(common[j])
        row = stats[frame]
        rows.append({
            "frame": frame,
            "baseline_translation_ape_m": float(errors["baseline"][j]),
            "adaptive_translation_ape_m": float(errors["adaptive_fallback"][j]),
            "provisional_translation_ape_m": float(errors["provisional"][j]),
            "tracking_inliers": int(row["tracking_inliers"]),
            "tracking_state": int(row["tracking_state"]),
            "fallback_used": int(row["fallback_used"]),
            "fallback_reason": int(row["fallback_reason"]),
            "static_left_features": int(row["static_left_features"]),
            "static_right_features": int(row["static_right_features"]),
            "fallback_left_features": int(row["fallback_left_features"]),
            "fallback_right_features": int(row["fallback_right_features"]),
            "stereo_matches": int(row["stereo_matches"]),
            "grid_coverage": float(row["grid_coverage"]),
            "static_map_points": int(row["static_map_points"]),
            "provisional_map_points": int(row["provisional_map_points"]),
            "promoted_map_points": int(row["promoted_map_points"]),
            "promotion_count": int(row["promotion_count"]),
            "rejection_count": int(row["rejection_count"]),
        })
    with (OUT / "ape_interval_5000_6500.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=180, sharex=True)
    for name, label in (("baseline", "Baseline"), ("adaptive_fallback", "Adaptive fallback"), ("provisional", "Provisional")):
        axes[0].plot(plot_frames, errors[name][selected], lw=1.1, label=label)
    axes[0].set_ylabel("Translation APE [m]"); axes[0].set_title("Translation APE expansion, frames 5000–6500")
    axes[0].axvspan(5500, 6000, color="#f59e0b", alpha=.10, label="APE growth interval")
    axes[1].plot(plot_frames, [r["tracking_inliers"] for r in rows], color="#16697a", lw=.9, label="Tracking inliers")
    axes[1].plot(plot_frames, [r["static_left_features"] + r["static_right_features"] for r in rows], color="#2e7d32", lw=.9, label="Static features (L+R)")
    axes[1].plot(plot_frames, [r["fallback_left_features"] + r["fallback_right_features"] for r in rows], color="#d97706", lw=.9, label="Fallback features (L+R)")
    axes[1].set_ylabel("count"); axes[1].set_title("Front-end support remains available, but static/fallback composition changes")
    axes[2].plot(plot_frames, [r["provisional_map_points"] for r in rows], color="#8e24aa", lw=.9, label="Provisional map points")
    axes[2].plot(plot_frames, [r["promoted_map_points"] for r in rows], color="#1976a3", lw=.9, label="Promoted map points")
    axes[2].plot(plot_frames, [r["static_map_points"] for r in rows], color="#455a64", lw=.9, label="Static map points")
    axes[2].set_ylabel("map points"); axes[2].set_xlabel("Frame index"); axes[2].set_title("MapPoint lifecycle in the same interval")
    for ax in axes: ax.grid(alpha=.25); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(OUT / "translation_ape_root_cause_5000_6500.png"); plt.close(fig)

    def summary(name, a, b):
        mask = (common >= a) & (common <= b)
        values = errors[name][mask]
        return float(np.sqrt(np.mean(values ** 2))), float(np.mean(values)), float(np.max(values))
    lines = [
        "# Provisional Translation APE 扩大根因分析（5000–6500 帧）", "",
        "## 数据与口径", "",
        "三组轨迹使用相同 GT、时间戳映射、共同帧和全局 SE(3) 对齐。区间内三组共同有效帧为 1501，provisional 状态全部为 OK。", "",
        "## 区间误差", "",
        "| 区间 | baseline RMSE | adaptive RMSE | provisional RMSE |", "|---|---:|---:|---:|",
    ]
    for a, b in ((5000, 5500), (5500, 6000), (6000, 6500), (5000, 6500)):
        vals = [summary(n, a, b)[0] for n in ("baseline", "adaptive_fallback", "provisional")]
        lines.append(f"| {a}–{b} | {vals[0]:.3f} m | {vals[1]:.3f} m | {vals[2]:.3f} m |")
    lines += ["", "## 证据链", "", "1. **不是特征选择差异**：provisional 与历史 adaptive fallback 使用同一融合掩码和同一硬回退配置，逐帧 static/fallback/stereo 计数一致；差异来自 MapPoint 生命周期和后端准入。", "2. **不是跟踪丢失**：5000–6500 帧全部为 `OK`，没有 LOST/RECENTLY_LOST；Translation APE 是连续漂移而非跟踪中断后的跳变。", "3. **前端退化不是主因**：区间平均内点为 provisional 173.1、adaptive 174.6 左右量级，Δ=1/10/100 的局部 RPE 仅有小幅差异；但 provisional 的全局 APE 在 5500–6000 达到约 4.924 m。", "4. **场景处于静态特征贫乏段**：5000–5500 回退触发约 81.2%，6000–6500 约 91.6%；静态特征分别约 194/198 和 166/174（左右目），大量观测依赖 Unknown 回退。", "5. **长期约束被削弱**：回退点在 provisional 版先进入 `PROVISIONAL`，未晋升前不进入 Global BA 和回环路径；同一批点在 adaptive 版则按普通 MapPoint 使用。provisional 日志没有 `*Loop detected`，而 adaptive 在约 3200 帧出现闭环检测；provisional 在约 2700 帧经历跟踪失败、建新地图，并在约 3400 帧发生 map merge。", "6. **生命周期规则造成候选点高流失**：provisional 版在该区间持续创建和淘汰候选点，promoted 数量从约 482 增至 631，而 provisional 数量在约 79–774 间波动；大量 Unknown 点达不到静态来源比例阈值，不能及时转为长期约束。", "7. **APE 后段回落的解释**：约 6100 帧后 APE 开始下降，同时 promoted/static map points 增加，说明局部建图和后续优化逐步补回约束；当前日志没有记录每次 BA 的修正量，因此这部分是有数据支持的推断，不是单独可证实的事件。", "", "## 最可能的根因", "", "`高回退比例/低静态特征` → `新增点大多为 provisional` → `晋升前不能参加 Global BA/闭环` → `长期地图约束不足且缺少闭环全局校正` → `在 5500–6000 帧累计平移漂移` → `Translation APE 扩大`。", "", "## 结论边界", "", "这是单次完整运行的根因定位。现有日志没有记录每帧 map id、BA 修正量、闭环候选数和逐点拒绝原因，因此不能把误差增长精确分解为某一个阈值的独立贡献；但“后端长期约束不足”由相同前端输入、全 OK 状态、相近局部 RPE、生命周期统计和闭环日志差异共同支持。", "", "图表见 `translation_ape_root_cause_5000_6500.png`，逐帧数据见 `ape_interval_5000_6500.csv`。"
    ]
    (OUT / "provisional_translation_ape_5000_6500根因分析.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved analysis to {OUT}")


if __name__ == "__main__":
    main()
