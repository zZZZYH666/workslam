#!/usr/bin/env python3
"""Summarize the complete A-E 7254-frame temporal ablation runs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/2026-09-29_w01_temporal_confidence"
OUT = BASE / "analysis_7254"
GT = ROOT / "data/W01_13Hz/GT_W01.txt"
TS = ROOT / "data/W01_13Hz/timestampSecNanoSec_W01.txt"
GROUPS = [
    ("A", "A_baseline_fixed", "20260929T113735Z_prefix7254_fixverify"),
    ("B", "B_adaptive_fallback", "20260929T120947Z_prefix7254"),
    ("C", "C_temporal_window", "20260929T122319Z_prefix7254"),
    ("D", "D_temporal_confidence", "20260929T123819Z_prefix7254"),
    ("E", "E_confidence_local_ba", "20260929T125148Z_prefix7254"),
]
ORIGINAL_BASELINE_RUNS = [
    BASE / "../2026-09-27_w01_geometric_stability/runs/baseline/run_01",
    BASE / "../2026-09-27_w01_geometric_stability/runs/baseline/run_02",
    BASE / "../2026-09-27_w01_geometric_stability/runs/baseline/run_03",
]


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def floats(values):
    return np.array([float(v) for v in values], dtype=float)


def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def load_ground_truth():
    timestamps = np.loadtxt(TS, ndmin=2)
    poses = np.loadtxt(GT, ndmin=2)
    ns = timestamps[:, 0] * 1e9 + timestamps[:, 1]
    xyz = poses[:, [3, 7, 11]]
    return ns, xyz


def rotation_from_quaternion(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx - qy * qy)],
    ])


def rigid_fit(source, target):
    if len(source) < 3:
        return np.eye(3), target.mean(axis=0) - source.mean(axis=0)
    smean, tmean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - smean).T @ (target - tmean))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return r, tmean - r @ smean


def trajectory_metrics(run: Path, gt_ns, gt_xyz, pattern="trajectory_map_*_semantic.csv"):
    map_files = sorted(run.glob(pattern))
    map_rows = []
    all_ate, all_ate_sq, all_rpe_t_sq, all_rpe_r_sq = [], [], [], []
    for path in map_files:
        data = rows(path)
        data = [row for row in data if int(float(row.get("frame", 0))) < len(gt_ns)]
        if not data:
            continue
        timestamps = np.array([int(float(r["timestamp_ns"])) for r in data])
        est_xyz = np.array([[float(r["tx"]), float(r["ty"]), float(r["tz"])] for r in data])
        gt_idx = np.searchsorted(gt_ns, timestamps).clip(0, len(gt_ns) - 1)
        left = np.maximum(gt_idx - 1, 0)
        gt_idx = np.where(np.abs(gt_ns[left] - timestamps) <= np.abs(gt_ns[gt_idx] - timestamps), left, gt_idx)
        target = gt_xyz[gt_idx]
        valid = np.array([int(r.get("pose_valid", "1")) == 1 for r in data])
        est_xyz, target = est_xyz[valid], target[valid]
        if len(est_xyz) == 0:
            continue
        rfit, tfit = rigid_fit(est_xyz, target)
        aligned = (rfit @ est_xyz.T).T + tfit
        ate = np.linalg.norm(aligned - target, axis=1)
        all_ate.extend(ate.tolist())
        all_ate_sq.extend((ate * ate).tolist())
        rpe_t, rpe_r = [], []
        rotations = [rotation_from_quaternion(float(x["qx"]), float(x["qy"]), float(x["qz"]), float(x["qw"])) for x in data if int(x.get("pose_valid", "1")) == 1]
        for i in range(1, len(aligned)):
            dt_est = aligned[i] - aligned[i - 1]
            dt_gt = target[i] - target[i - 1]
            rpe_t.append(float(np.linalg.norm(dt_est - dt_gt)))
            rel = rotations[i - 1].T @ rotations[i]
            angle = math.acos(float(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
            rpe_r.append(angle)
        all_rpe_t_sq.extend((np.array(rpe_t) ** 2).tolist())
        all_rpe_r_sq.extend((np.array(rpe_r) ** 2).tolist())
        map_rows.append({"map": path.name, "frames": len(est_xyz), "ate_rmse_m": float(np.sqrt(np.mean(ate * ate))), "rpe_trans_rmse_m": float(np.sqrt(np.mean(np.array(rpe_t) ** 2))) if rpe_t else float("nan"), "rpe_rot_rmse_deg": math.degrees(math.sqrt(float(np.mean(np.array(rpe_r) ** 2)))) if rpe_r else float("nan")})
    return map_rows, {
        "map_count": len(map_rows),
        "ate_rmse_m": math.sqrt(float(np.mean(all_ate_sq))) if all_ate_sq else float("nan"),
        "ate_mean_m": float(np.mean(all_ate)) if all_ate else float("nan"),
        "ate_p95_m": float(np.percentile(all_ate, 95)) if all_ate else float("nan"),
        "cumulative_ape_m": float(np.sum(all_ate)) if all_ate else float("nan"),
        "cumulative_ate_m": float(np.sum(all_ate)) if all_ate else float("nan"),
        "rpe_trans_rmse_m": math.sqrt(float(np.mean(all_rpe_t_sq))) if all_rpe_t_sq else float("nan"),
        "rpe_rot_rmse_deg": math.degrees(math.sqrt(float(np.mean(all_rpe_r_sq)))) if all_rpe_r_sq else float("nan"),
    }


def analyze(letter, group, run_id, gt_ns, gt_xyz):
    run = BASE / group / run_id
    stats, times, lifecycle, backend = (rows(run / name) for name in ("frame_stats_semantic.csv", "frame_times_semantic.csv", "semantic_lifecycle_events.csv", "backend_interval_stats.csv"))
    n = len(stats)
    state_counts = {state: sum(row.get("tracking_state") == state for row in stats) for state in ("0", "1", "2", "3", "4")}
    track_ms = [float(row["track_time_ms"]) for row in stats if row.get("track_time_ms")]
    alpha = [float(row["fallback_alpha"]) for row in stats]
    risk = [float(row["fallback_risk"]) for row in stats]
    fallback = [row for row in stats if row.get("fallback_used") == "1"]
    reasons = {}
    for row in lifecycle:
        reasons[row.get("reason", "")] = reasons.get(row.get("reason", ""), 0) + 1
    confidence = [float(row["confidence"]) for row in lifecycle if row.get("confidence", "")]
    summary = rows(run / "run_summary.csv")[0]
    maps, traj = trajectory_metrics(run, gt_ns, gt_xyz)
    result = {
        "group": letter, "run_id": run_id, "frames": n, "tracking_success_rate": state_counts["2"] / n,
        "recently_lost_frames": state_counts["3"], "lost_frames": state_counts["4"], "tracking_state_0": state_counts["0"], "tracking_state_1": state_counts["1"],
        "fallback_frame_ratio": len(fallback) / n, "fallback_mean_risk": float(np.mean(risk)), "fallback_p95_risk": percentile(risk, 95), "fallback_mean_alpha": float(np.mean(alpha)), "fallback_max_alpha": max(alpha),
        "track_mean_ms": float(np.mean(track_ms)), "track_p95_ms": percentile(track_ms, 95), "runtime_sec": float(np.sum(track_ms) / 1000.0),
        "atlas_maps": int(summary["atlas_maps"]), "atlas_points": int(summary["current_map_points"]), "static_map_points": int(summary["static_map_points"]), "provisional_map_points": int(summary["provisional_map_points"]), "promoted_map_points": int(summary["promoted_map_points"]), "promotion_count": int(summary["promotion_count"]), "rejection_count": int(summary["rejection_count"]),
        "lifecycle_events": len(lifecycle), "promotion_events": reasons.get("promotion", 0), "rejection_events": sum(v for k, v in reasons.items() if "reject" in k or "dynamic" in k or "expired" in k), "confidence_mean": float(np.mean(confidence)) if confidence else float("nan"),
        **traj,
    }
    return result, maps, reasons


def analyze_original_baseline(gt_ns, gt_xyz):
    """Aggregate the three original baseline repeats on their first 7254 frames."""
    per_run = []
    map_rows = []
    for run in ORIGINAL_BASELINE_RUNS:
        stats = [row for row in rows(run / "frame_stats_baseline.csv") if int(row["index"]) < 7254]
        times = [float(row["track_time_ms"]) for row in stats]
        states = [row.get("tracking_state", "") for row in stats]
        maps, traj = trajectory_metrics(run, gt_ns, gt_xyz, "trajectory_map_*_baseline.csv")
        per_run.append({
            "frames": len(stats),
            "tracking_success_rate": states.count("2") / len(stats),
            "recently_lost_frames": states.count("3"), "lost_frames": states.count("4"),
            "tracking_state_0": states.count("0"), "tracking_state_1": states.count("1"),
            "track_mean_ms": float(np.mean(times)), "track_p95_ms": percentile(times, 95),
            "runtime_sec": float(np.sum(times) / 1000.0),
            **traj,
        })
        map_rows.extend(item for item in maps)

    def mean(key):
        return float(np.mean([item[key] for item in per_run]))

    def std(key):
        return float(np.std([item[key] for item in per_run], ddof=1))

    result = {
        "group": "F-original", "run_id": "baseline_run_01..03_mean",
        "replicates": len(per_run), "frames": int(mean("frames")),
        "tracking_success_rate": mean("tracking_success_rate"),
        "tracking_success_rate_std": std("tracking_success_rate"),
        "recently_lost_frames": mean("recently_lost_frames"), "lost_frames": mean("lost_frames"),
        "tracking_state_0": mean("tracking_state_0"), "tracking_state_1": mean("tracking_state_1"),
        "fallback_frame_ratio": None, "fallback_mean_risk": None, "fallback_p95_risk": None,
        "fallback_mean_alpha": None, "fallback_max_alpha": None,
        "track_mean_ms": mean("track_mean_ms"), "track_mean_ms_std": std("track_mean_ms"),
        "track_p95_ms": mean("track_p95_ms"), "atlas_maps": 1,
        "runtime_sec": mean("runtime_sec"),
        "atlas_points": None, "static_map_points": None, "provisional_map_points": None,
        "promoted_map_points": None, "promotion_count": None, "rejection_count": None,
        "lifecycle_events": None, "promotion_events": None, "rejection_events": None,
        "confidence_mean": None, "map_count": 1,
        "ate_rmse_m": mean("ate_rmse_m"), "ate_rmse_m_std": std("ate_rmse_m"),
        "ate_mean_m": mean("ate_mean_m"), "ate_p95_m": mean("ate_p95_m"),
        "cumulative_ape_m": mean("cumulative_ape_m"), "cumulative_ate_m": mean("cumulative_ate_m"),
        "rpe_trans_rmse_m": mean("rpe_trans_rmse_m"), "rpe_rot_rmse_deg": mean("rpe_rot_rmse_deg"),
    }
    return result, map_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt_ns, gt_xyz = load_ground_truth()
    results, map_rows, reason_rows = [], [], []
    for letter, group, run_id in GROUPS:
        result, maps, reasons = analyze(letter, group, run_id, gt_ns[:7254], gt_xyz[:7254])
        results.append(result)
        for row in maps:
            map_rows.append({"group": letter, **row})
        for reason, count in sorted(reasons.items()):
            reason_rows.append({"group": letter, "reason": reason, "count": count})
    baseline_result, baseline_maps = analyze_original_baseline(gt_ns[:7254], gt_xyz[:7254])
    results.insert(0, baseline_result)
    for item in baseline_maps:
        map_rows.append({"group": "F-original", **item})
    fields = list(results[0])
    with (OUT / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    with (OUT / "map_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(map_rows[0]))
        writer.writeheader(); writer.writerows(map_rows)
    with (OUT / "lifecycle_reasons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "reason", "count"])
        writer.writeheader(); writer.writerows(reason_rows)
    (OUT / "original_baseline_provenance.json").write_text(json.dumps({
        "group": "F-original",
        "source_runs": [str(path.relative_to(ROOT)) for path in ORIGINAL_BASELINE_RUNS],
        "source_config": "configs/finnforest_w01_stereo_rectified.yaml",
        "source_frame_count": 9210,
        "reported_frame_count": 7254,
        "aggregation": "mean_of_three_repeats",
        "metrics": ["tracking", "track_time", "per_map_se3_trajectory", "rpe"],
        "not_comparable_fields": ["fallback", "risk", "temporal_confidence", "lifecycle", "map_points_at_7254"],
    }, indent=2) + "\n")
    def display(value, digits=4):
        return "N/A" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}f}"

    lines = ["# 7254 帧原始 baseline + A-E 指标分析", "", "F-original 为 2026-09-27 原始 baseline 的 3 次重复实验，截取前 7254 帧后取均值；它不是当前 A 组。轨迹误差按每个 atlas map 独立 SE(3) 对齐并计算，没有跨 map 拼接。", "", "## 主指标", "", "|组|Tracking 成功率|RECENTLY_LOST|LOST|Fallback 比例|平均风险|平均耗时 ms|P95 ms|Atlas maps|ATE RMSE m|ATE P95 m|RPE 平移 RMSE m|RPE 旋转 RMSE deg|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"|{r['group']}|{display(r['tracking_success_rate'])}|{display(r['recently_lost_frames'], 1)}|{display(r['lost_frames'], 1)}|{display(r['fallback_frame_ratio'])}|{display(r['fallback_mean_risk'])}|{display(r['track_mean_ms'], 2)}|{display(r['track_p95_ms'], 2)}|{display(r['atlas_maps'], 0)}|{display(r['ate_rmse_m'])}|{display(r['ate_p95_m'])}|{display(r['rpe_trans_rmse_m'])}|{display(r['rpe_rot_rmse_deg'], 3)}|")
    lines += ["", "## 地图生命周期", "", "|组|最终地图点|Static|Provisional|Promoted|Promotion count|Rejection count|Lifecycle events|平均 confidence|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"|{r['group']}|{display(r['atlas_points'], 0)}|{display(r['static_map_points'], 0)}|{display(r['provisional_map_points'], 0)}|{display(r['promoted_map_points'], 0)}|{display(r['promotion_count'], 0)}|{display(r['rejection_count'], 0)}|{display(r['lifecycle_events'], 0)}|{display(r['confidence_mean'])}|")
    lines += [
        "", "## A-E 汇总表", "",
        "ATE 使用绝对轨迹误差 RMSE；APE 是逐帧平移绝对位姿误差，APE RMSE 与 ATE RMSE 在本表中数值相同，APE P95 用于表示误差尾部。Cumulative ATE/APE 是整段序列逐帧平移误差的累加值（Σ error）；当前实现只统计平移，因此两者数值相同。RPE 使用平移 RPE RMSE；Runtime 为所有帧 `track_time` 的总和，不包含进程启动和关闭开销。",
        "", "| Group | Frames | Tracking success | RECENTLY_LOST | LOST | Fallback ratio | Mean risk | ATE RMSE (m) | APE RMSE (m) | APE P95 (m) | Cumulative ATE (m) | Cumulative APE (m) | RPE RMSE (m) | Promotions | Rejections | Map points | Mean frame (ms) | Runtime (s) |", "| ----- | -----: | ---------------: | -------------: | ---: | --------------: | ---------: | ------------: | ------------: | -----------: | -----------------: | -----------------: | -----------: | ---------: | ---------: | ---------: | --------------: | -----------: |",
    ]
    for r in results:
        if r["group"] == "F-original":
            continue
        lines.append(f"| {r['group']} | {display(r['frames'], 0)} | {display(r['tracking_success_rate'] * 100, 2)}% | {display(r['recently_lost_frames'], 0)} | {display(r['lost_frames'], 0)} | {display(r['fallback_frame_ratio'] * 100, 2)}% | {display(r['fallback_mean_risk'])} | {display(r['ate_rmse_m'])} | {display(r['ate_rmse_m'])} | {display(r['ate_p95_m'])} | {display(r['cumulative_ate_m'])} | {display(r['cumulative_ape_m'])} | {display(r['rpe_trans_rmse_m'])} | {display(r['promotion_count'], 0)} | {display(r['rejection_count'], 0)} | {display(r['atlas_points'], 0)} | {display(r['track_mean_ms'], 2)} | {display(r['runtime_sec'], 2)} |")
    lines += [
        "", "## 解释", "",
        "F-original 来自 `results/2026-09-27_w01_geometric_stability/runs/baseline/run_01..03`，使用原始 baseline 配置，三次重复实验均截取前 7254 帧后取均值。它是原始系统参考，不是当前 A 组；A 仍然启用了固定 semantic fallback 和 Provisional 生命周期。",
        "",
        "原始 baseline 没有 semantic fallback、风险、观测窗口、confidence 或生命周期 CSV，因此这些字段显示为 N/A。地图点数量也不使用 9210 帧最终值，避免与 7254 帧结果混淆。",
        "",
        "完整机器可读数据：`metrics.csv`、`map_metrics.csv`、`lifecycle_reasons.csv`。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
