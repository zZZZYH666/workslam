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


def trajectory_metrics(run: Path, gt_ns, gt_xyz):
    map_files = sorted(run.glob("trajectory_map_*_semantic.csv"))
    map_rows = []
    all_ate_sq, all_rpe_t_sq, all_rpe_r_sq = [], [], []
    for path in map_files:
        data = rows(path)
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
        "track_mean_ms": float(np.mean(track_ms)), "track_p95_ms": percentile(track_ms, 95),
        "atlas_maps": int(summary["atlas_maps"]), "atlas_points": int(summary["current_map_points"]), "static_map_points": int(summary["static_map_points"]), "provisional_map_points": int(summary["provisional_map_points"]), "promoted_map_points": int(summary["promoted_map_points"]), "promotion_count": int(summary["promotion_count"]), "rejection_count": int(summary["rejection_count"]),
        "lifecycle_events": len(lifecycle), "promotion_events": reasons.get("promotion", 0), "rejection_events": sum(v for k, v in reasons.items() if "reject" in k or "dynamic" in k or "expired" in k), "confidence_mean": float(np.mean(confidence)) if confidence else float("nan"),
        **traj,
    }
    return result, maps, reasons


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
    lines = ["# 7254 帧 A-E 指标分析", "", "轨迹误差按每个 atlas map 独立 SE(3) 对齐并计算，再汇总各 map；没有跨 map 拼接。", "", "## 主指标", "", "|组|Tracking 成功率|RECENTLY_LOST|LOST|Fallback 比例|平均风险|平均耗时 ms|P95 ms|Atlas maps|ATE RMSE m|RPE 平移 RMSE m|RPE 旋转 RMSE deg|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"|{r['group']}|{r['tracking_success_rate']:.4f}|{r['recently_lost_frames']}|{r['lost_frames']}|{r['fallback_frame_ratio']:.4f}|{r['fallback_mean_risk']:.4f}|{r['track_mean_ms']:.2f}|{r['track_p95_ms']:.2f}|{r['atlas_maps']}|{r['ate_rmse_m']:.4f}|{r['rpe_trans_rmse_m']:.4f}|{r['rpe_rot_rmse_deg']:.3f}|")
    lines += ["", "## 地图生命周期", "", "|组|最终地图点|Static|Provisional|Promoted|Promotion count|Rejection count|Lifecycle events|平均 confidence|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"|{r['group']}|{r['atlas_points']}|{r['static_map_points']}|{r['provisional_map_points']}|{r['promoted_map_points']}|{r['promotion_count']}|{r['rejection_count']}|{r['lifecycle_events']}|{r['confidence_mean']:.4f}|")
    lines += ["", "完整机器可读数据：`metrics.csv`、`map_metrics.csv`、`lifecycle_reasons.csv`。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
