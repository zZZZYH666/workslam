#!/usr/bin/env python3
"""Recover a semantic-projection result from an existing pair of YOLO masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from semantic_projection_sgbm import (
    _cycle_errors,
    _disparity_view,
    _make_visualization,
    _resize_mask,
    _write_disparity,
    _write_mask,
    build_sgbm,
    fuse_labels,
    project_labels,
)


def frame_indices(dataset: Path) -> list[int]:
    left = {int(p.stem) for p in (dataset / "images_cam2_sr22555667").glob("*.png") if p.stem.isdigit()}
    right = {int(p.stem) for p in (dataset / "images_cam3_sr22555660").glob("*.png") if p.stem.isdigit()}
    return sorted(left & right)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/W01_13Hz")
    p.add_argument("--mask-input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="YOLO/yolo_workspace/train_my/runs/segment/yolov8s-seg-base/weights/best.pt")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--num-disparities", type=int, default=256)
    p.add_argument("--lr-threshold", type=float, default=2.0)
    p.add_argument("--support-window", type=int, default=3)
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--dominance-ratio", type=float, default=0.6)
    p.add_argument("--visualization-first", type=int, default=10)
    p.add_argument("--visualization-every", type=int, default=100)
    p.add_argument("--visualization-scale", type=float, default=0.35)
    args = p.parse_args()

    dataset, mask_input, output = map(Path, (args.dataset, args.mask_input, args.output))
    left_dir, right_dir = dataset / "images_cam2_sr22555667", dataset / "images_cam3_sr22555660"
    frames = frame_indices(dataset)
    sample = cv2.imread(str(left_dir / f"{frames[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if sample is None:
        raise RuntimeError("failed to read sample frame")
    height, width = sample.shape
    work_shape = (round(height * args.scale), round(width * args.scale))
    left_matcher = build_sgbm(args.scale, "left", args.num_disparities)
    right_matcher = build_sgbm(args.scale, "right", args.num_disparities)
    for sub in ("masks/cam2", "masks/cam3", "projected_masks/cam2", "projected_masks/cam3",
                "fused_masks/cam2", "fused_masks/cam3", "disparity/cam2_to_cam3",
                "disparity/cam3_to_cam2", "visualizations"):
        (output / sub).mkdir(parents=True, exist_ok=True)

    fields = ["index", "left_valid_percent", "right_valid_percent", "left_projected_pixels", "right_projected_pixels",
              "left_fused_pixels", "right_fused_pixels", "left_conflict_pixels", "right_conflict_pixels",
              "cycle_left_median_px", "cycle_right_median_px", "yolo_time_ms", "sgbm_time_ms", "total_time_ms"]
    rows = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for n, frame_id in enumerate(frames):
            stem = f"{frame_id:06d}"
            left = cv2.imread(str(left_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(right_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            left_mask = cv2.imread(str(mask_input / "masks_cam2" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            right_mask = cv2.imread(str(mask_input / "masks_cam3" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            if any(x is None for x in (left, right, left_mask, right_mask)):
                raise RuntimeError(f"missing input for frame {frame_id}")
            left_work, right_work = (cv2.resize(x, (work_shape[1], work_shape[0]), interpolation=cv2.INTER_AREA) for x in (left, right))
            left_mask_work = _resize_mask(left_mask, work_shape)
            right_mask_work = _resize_mask(right_mask, work_shape)
            t = time.perf_counter()
            lf = pool.submit(left_matcher.compute, left_work, right_work)
            rf = pool.submit(right_matcher.compute, right_work, left_work)
            disp_left = lf.result().astype(np.float32) / 16.0
            disp_right = rf.result().astype(np.float32) / 16.0
            sgbm_ms = (time.perf_counter() - t) * 1000.0
            to_right, right_stats, _ = project_labels(left_mask_work, disp_left, disp_right, "left_to_right", args.lr_threshold, args.support_window, args.min_support, args.dominance_ratio, args.num_disparities)
            to_left, left_stats, _ = project_labels(right_mask_work, disp_right, disp_left, "right_to_left", args.lr_threshold, args.support_window, args.min_support, args.dominance_ratio, args.num_disparities)
            left_projected, right_projected = _resize_mask(to_left, left.shape), _resize_mask(to_right, right.shape)
            left_fused, left_fuse_stats = fuse_labels(left_mask, left_projected)
            right_fused, right_fuse_stats = fuse_labels(right_mask, right_projected)
            _write_mask(output / "masks/cam2" / f"{stem}.png", left_mask)
            _write_mask(output / "masks/cam3" / f"{stem}.png", right_mask)
            _write_mask(output / "projected_masks/cam2" / f"{stem}.png", left_projected)
            _write_mask(output / "projected_masks/cam3" / f"{stem}.png", right_projected)
            _write_mask(output / "fused_masks/cam2" / f"{stem}.png", left_fused)
            _write_mask(output / "fused_masks/cam3" / f"{stem}.png", right_fused)
            _write_disparity(output / "disparity/cam2_to_cam3" / f"{stem}.png", disp_left)
            _write_disparity(output / "disparity/cam3_to_cam2" / f"{stem}.png", disp_right)
            if frame_id < args.visualization_first or (args.visualization_every > 0 and frame_id % args.visualization_every == 0):
                _write_mask(output / "visualizations" / f"frame_{stem}.png", _make_visualization(left, right, left_fused, right_fused, disp_left, disp_right, args.visualization_scale))
                _write_mask(output / "visualizations" / f"disparity_cam2_to_cam3_{stem}.png", _disparity_view(disp_left))
                _write_mask(output / "visualizations" / f"disparity_cam3_to_cam2_{stem}.png", _disparity_view(disp_right))
            valid_left = np.isfinite(disp_left) & (disp_left > 0)
            valid_right = np.isfinite(disp_right) & (disp_right < 0) & (disp_right > -args.num_disparities)
            cycle_left, cycle_right = _cycle_errors(disp_left, disp_right, "left_to_right", args.num_disparities), _cycle_errors(disp_right, disp_left, "right_to_left", args.num_disparities)
            rows.append({"index": frame_id, "left_valid_percent": 100 * np.count_nonzero(valid_left) / valid_left.size, "right_valid_percent": 100 * np.count_nonzero(valid_right) / valid_right.size,
                         "left_projected_pixels": int(np.count_nonzero(left_projected)), "right_projected_pixels": int(np.count_nonzero(right_projected)),
                         "left_fused_pixels": int(np.count_nonzero(left_fused)), "right_fused_pixels": int(np.count_nonzero(right_fused)),
                         "left_conflict_pixels": left_stats["conflict_pixels"], "right_conflict_pixels": right_stats["conflict_pixels"],
                         "cycle_left_median_px": float(np.median(cycle_left)) if cycle_left.size else float("nan"), "cycle_right_median_px": float(np.median(cycle_right)) if cycle_right.size else float("nan"),
                         "yolo_time_ms": 0.0, "sgbm_time_ms": sgbm_ms, "total_time_ms": 0.0})
            if (n + 1) % 100 == 0:
                print(f"recovered {n + 1}/{len(frames)}", flush=True)
    with (output / "summary.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    metadata = {"dataset": str(dataset.resolve()), "model": str(Path(args.model).resolve()), "model_sha256": sha256(Path(args.model)), "model_names": {"0": "road", "1": "tree"}, "frames_total": len(frames), "frame_start": frames[0], "frame_end_exclusive": frames[-1] + 1,
                "image_shape": [height, width], "scale": args.scale, "sgbm": {"num_disparities": args.num_disparities, "block_size": 5, "mode": "SGBM_3WAY"},
                "disparity_storage": {"format": "uint16_png", "scale": 16, "offset": 32768, "invalid": 0, "shape": list(work_shape)},
                "projection": {"cycle_threshold_px": args.lr_threshold, "support_window": args.support_window, "min_support": args.min_support, "dominance_ratio": args.dominance_ratio, "fusion": "preserve_target_nonunknown", "single_pass": True},
                "visualizations": {"first_frames": args.visualization_first, "every_n_frames": args.visualization_every, "scale": args.visualization_scale}, "videos": False, "recovered_from_masks": str(mask_input.resolve()), "elapsed_seconds": time.perf_counter() - start}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
