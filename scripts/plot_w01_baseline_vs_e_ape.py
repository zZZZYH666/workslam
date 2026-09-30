#!/usr/bin/env python3
"""Compare the 7254-frame E run with the original baseline using APE curves.

Baseline is aggregated over three independent runs.  E is one run with three
atlas maps; each map is aligned independently before its local APE is placed
on the global frame axis.  This matches the multimap evaluation convention
used by the temporal-confidence report and does not claim a cross-map global
trajectory error.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FRAME_COUNT = 7254
LAST_FRAME = FRAME_COUNT - 1
GT_DIR = ROOT / "data/W01_13Hz"
BASELINE_DIR = ROOT / "results/2026-09-27_w01_geometric_stability/runs/baseline"
E_RUN = ROOT / "results/2026-09-29_w01_temporal_confidence/E_confidence_local_ba/20260929T125148Z_prefix7254"
OUT = ROOT / "results/2026-09-29_w01_temporal_confidence/analysis_7254/comparison"


def load_timestamps(path: Path) -> np.ndarray:
    rows = np.atleast_2d(np.loadtxt(path))[:FRAME_COUNT]
    return np.rint(rows[:, 0] * 1.0e9 + rows[:, 1]).astype(np.int64)


def load_groundtruth(path: Path) -> np.ndarray:
    rows = np.atleast_2d(np.loadtxt(path))[:FRAME_COUNT]
    poses = np.repeat(np.eye(4)[None, :, :], len(rows), axis=0)
    poses[:, :3, :] = rows.reshape(-1, 3, 4)
    return poses


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, target_mean - rotation @ source_mean


def map_errors(path: Path, groundtruth: np.ndarray):
    rows = [r for r in read_rows(path) if int(r["frame"]) <= LAST_FRAME and r.get("pose_valid", "1") == "1"]
    if len(rows) < 3:
        raise ValueError(f"not enough valid poses for alignment: {path}")
    frames = np.asarray([int(r["frame"]) for r in rows], dtype=int)
    positions = np.asarray([[float(r["tx"]), float(r["ty"]), float(r["tz"])] for r in rows])
    rotations = np.asarray([
        quaternion_matrix(float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"]))
        for r in rows
    ])
    target = groundtruth[frames]
    fit_rotation, fit_translation = rigid_fit(positions, target[:, :3, 3])
    aligned_positions = (fit_rotation @ positions.T).T + fit_translation
    aligned_rotations = np.einsum("ij,njk->nik", fit_rotation, rotations)
    translation = np.linalg.norm(aligned_positions - target[:, :3, 3], axis=1)
    relative_rotation = np.einsum("nij,njk->nik", np.transpose(target[:, :3, :3], (0, 2, 1)), aligned_rotations)
    cosine = np.clip((np.trace(relative_rotation, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation = np.degrees(np.arccos(cosine))
    return frames, translation, rotation


def collect_baseline(groundtruth):
    curves = []
    for run in ("run_01", "run_02", "run_03"):
        frames, translation, rotation = map_errors(BASELINE_DIR / run / "trajectory_map_0_baseline.csv", groundtruth)
        t = np.full(FRAME_COUNT, np.nan)
        r = np.full(FRAME_COUNT, np.nan)
        t[frames], r[frames] = translation, rotation
        curves.append((t, r))
    return curves


def collect_e(groundtruth):
    translation = np.full(FRAME_COUNT, np.nan)
    rotation = np.full(FRAME_COUNT, np.nan)
    for path in sorted(E_RUN.glob("trajectory_map_*_semantic.csv")):
        frames, current_t, current_r = map_errors(path, groundtruth)
        translation[frames], rotation[frames] = current_t, current_r
    return [(translation, rotation)]


def mean_sd(curves):
    values_t = np.asarray([item[0] for item in curves])
    values_r = np.asarray([item[1] for item in curves])
    count_t = np.sum(np.isfinite(values_t), axis=0)
    count_r = np.sum(np.isfinite(values_r), axis=0)
    mean_t = np.full(FRAME_COUNT, np.nan)
    mean_r = np.full(FRAME_COUNT, np.nan)
    valid_t = count_t > 0
    valid_r = count_r > 0
    mean_t[valid_t] = np.nanmean(values_t[:, valid_t], axis=0)
    mean_r[valid_r] = np.nanmean(values_r[:, valid_r], axis=0)
    sd_t = np.nanstd(values_t, axis=0, ddof=1) if len(curves) > 1 else np.zeros(FRAME_COUNT)
    sd_r = np.nanstd(values_r, axis=0, ddof=1) if len(curves) > 1 else np.zeros(FRAME_COUNT)
    sd_t[count_t == 0] = np.nan
    sd_r[count_r == 0] = np.nan
    return mean_t, sd_t, mean_r, sd_r, count_t, count_r


def finite_stats(curves):
    translations = np.concatenate([t[np.isfinite(t)] for t, _ in curves])
    rotations = np.concatenate([r[np.isfinite(r)] for _, r in curves])
    return {
        "runs_or_segments": len(curves),
        "valid_frames": int(len(translations)),
        "translation_ape_rmse_m": float(np.sqrt(np.mean(translations ** 2))),
        "translation_ape_p95_m": float(np.percentile(translations, 95)),
        "translation_ape_cumulative_m": float(np.sum(translations)),
        "rotation_ape_rmse_deg": float(np.sqrt(np.mean(rotations ** 2))),
        "rotation_ape_p95_deg": float(np.percentile(rotations, 95)),
    }


def write_curve_csv(baseline, e):
    bt, bs, br, brs, btn, brn = mean_sd(baseline)
    et, es, er, ers, etn, ern = mean_sd(e)
    path = OUT / "baseline_vs_E_ape_curves.csv"
    with path.open("w", newline="") as stream:
        fields = [
            "frame", "baseline_n", "baseline_translation_mean_m", "baseline_translation_sd_m",
            "baseline_rotation_mean_deg", "baseline_rotation_sd_deg", "E_n",
            "E_translation_mean_m", "E_translation_sd_m", "E_rotation_mean_deg", "E_rotation_sd_deg",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame in range(FRAME_COUNT):
            writer.writerow({
                "frame": frame,
                "baseline_n": int(btn[frame]),
                "baseline_translation_mean_m": bt[frame],
                "baseline_translation_sd_m": bs[frame],
                "baseline_rotation_mean_deg": br[frame],
                "baseline_rotation_sd_deg": brs[frame],
                "E_n": int(etn[frame]),
                "E_translation_mean_m": et[frame],
                "E_translation_sd_m": es[frame],
                "E_rotation_mean_deg": er[frame],
                "E_rotation_sd_deg": ers[frame],
            })
    return path


def plot_curves(baseline, e):
    bt, bs, br, brs, _, _ = mean_sd(baseline)
    et, es, er, ers, _, _ = mean_sd(e)
    x = np.arange(FRAME_COUNT)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=180, sharex=True)
    for ax, baseline_mean, baseline_sd, e_mean, e_sd, ylabel in (
        (axes[0], bt, bs, et, es, "Translation APE [m]"),
        (axes[1], br, brs, er, ers, "Rotation APE [deg]"),
    ):
        ax.plot(x, baseline_mean, color="#555555", lw=1.5, label="ORB-SLAM3 baseline (n=3)")
        ax.fill_between(x, baseline_mean - baseline_sd, baseline_mean + baseline_sd, color="#555555", alpha=0.18)
        ax.plot(x, e_mean, color="#1769aa", lw=1.4, label="E: confidence + Local BA (n=1)")
        ax.fill_between(x, e_mean - e_sd, e_mean + e_sd, color="#1769aa", alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    for frame in (323, 7245):
        axes[0].axvline(frame, color="#1769aa", lw=0.8, ls="--", alpha=0.35)
        axes[1].axvline(frame, color="#1769aa", lw=0.8, ls="--", alpha=0.35)
    axes[1].set_xlabel("Frame")
    axes[0].set_title("W01 prefix-7254 APE: original baseline vs E\nMean +/- 1 SD for baseline; E uses independent local alignment per atlas map")
    fig.tight_layout()
    output = OUT / "baseline_vs_E_ape_curves_mean_sd.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def write_report(baseline, e):
    summary = {
        "frame_count": FRAME_COUNT,
        "baseline": finite_stats(baseline),
        "E": finite_stats(e),
        "alignment": "independent SE(3) alignment per exported atlas map",
        "E_run": str(E_RUN.relative_to(ROOT)),
        "baseline_runs": [str((BASELINE_DIR / run).relative_to(ROOT)) for run in ("run_01", "run_02", "run_03")],
        "caveat": "E has one run and three atlas maps; its shaded SD is zero and its APE is local per map, not a cross-map global APE.",
    }
    (OUT / "baseline_vs_E_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# E 与原始 baseline 的 7254 帧 APE 对比", "",
        "图形沿用 `baseline_vs_turn_fallback_ape_curves_mean_sd.png` 的均值 +/- 标准差形式。Baseline 使用 3 次原始单地图运行；E 使用当前 7254 帧单次运行。",
        "", "## 对齐和限制", "",
        "E 的 3 个 atlas map 分别进行 SE(3) 对齐后放回原始帧号，因此图中是多地图局部 APE，不是跨 atlas map 拼接的全局 APE。E 只有 1 次运行，所以其标准差带为 0；蓝色虚线为 E 的 map 切换帧 323 和 7245。切换附近的尖峰主要反映独立局部对齐，尤其是只有少量有效帧的末端 map，不应直接解释为全局轨迹跳变。", "",
        "## 指标", "",
        "| Group | Valid frames | Translation APE RMSE (m) | Translation APE P95 (m) | Cumulative translation APE (m) | Rotation APE RMSE (deg) | Rotation APE P95 (deg) |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, curves in (("Baseline", baseline), ("E", e)):
        s = finite_stats(curves)
        lines.append(f"| {label} | {s['valid_frames']} | {s['translation_ape_rmse_m']:.4f} | {s['translation_ape_p95_m']:.4f} | {s['translation_ape_cumulative_m']:.4f} | {s['rotation_ape_rmse_deg']:.4f} | {s['rotation_ape_p95_deg']:.4f} |")
    lines += ["", "## 输出", "", "- `baseline_vs_E_ape_curves_mean_sd.png`：APE 曲线。", "- `baseline_vs_E_ape_curves.csv`：逐帧均值、标准差和有效样本数。", "- `baseline_vs_E_summary.json`：数据来源和汇总指标。"]
    (OUT / "baseline_vs_E_report.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timestamps = load_timestamps(GT_DIR / "timestampSecNanoSec_W01.txt")
    groundtruth = load_groundtruth(GT_DIR / "GT_W01.txt")
    if len(timestamps) != FRAME_COUNT or len(groundtruth) != FRAME_COUNT:
        raise ValueError("ground truth/timestamp frame count is not 7254")
    baseline = collect_baseline(groundtruth)
    e = collect_e(groundtruth)
    write_curve_csv(baseline, e)
    output = plot_curves(baseline, e)
    write_report(baseline, e)
    print(output)


if __name__ == "__main__":
    main()
