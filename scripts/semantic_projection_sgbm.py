#!/usr/bin/env python3
"""Generate bidirectionally fused semantic masks for FinnForest W01.

The implementation intentionally keeps semantic fusion independent from
ORB-SLAM3.  YOLO produces the per-camera static labels and StereoSGBM is used
only to transfer labels through valid, left-right-consistent pixels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


LABEL_UNKNOWN = 0
LABEL_TREE = 1
LABEL_ROAD = 2


def build_sgbm(scale: float, direction: str, num_disparities: int = 256) -> cv2.StereoSGBM:
    if direction not in {"left", "right"}:
        raise ValueError("direction must be left or right")
    if num_disparities <= 0 or num_disparities % 16:
        raise ValueError("num_disparities must be a positive multiple of 16")
    block = 5
    channels = 1
    p1 = 8 * channels * block * block
    p2 = 32 * channels * block * block
    minimum = 0 if direction == "left" else -num_disparities
    return cv2.StereoSGBM_create(
        minDisparity=minimum,
        numDisparities=num_disparities,
        blockSize=block,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(np.uint8, copy=False)
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def valid_disparity(disparity: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return np.isfinite(disparity) & (disparity >= minimum) & (disparity <= maximum)


def project_labels(
    source_labels: np.ndarray,
    source_disparity: np.ndarray,
    reverse_disparity: np.ndarray,
    direction: str,
    cycle_threshold: float = 2.0,
    support_window: int = 3,
    min_support: int = 3,
    dominance_ratio: float = 0.6,
    num_disparities: int = 256,
) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
    """Project source labels and return accepted labels plus diagnostics.

    The reverse disparity has the opposite sign convention.  Candidates are
    accumulated at their nearest target pixel, then accepted only when a
    consistent local majority exists.  The returned candidate image contains
    only accepted projected labels; zero means no candidate.
    """
    if source_labels.shape != source_disparity.shape or source_labels.shape != reverse_disparity.shape:
        raise ValueError("labels and disparities must have identical shapes")
    if support_window % 2 != 1 or support_window < 1:
        raise ValueError("support_window must be a positive odd number")
    if direction not in {"left_to_right", "right_to_left"}:
        raise ValueError("invalid projection direction")

    height, width = source_labels.shape
    candidates = np.zeros((height, width), dtype=np.uint8)
    votes = np.zeros((height, width, 3), dtype=np.uint16)
    source_valid = np.isfinite(source_disparity)
    source_valid &= source_labels != LABEL_UNKNOWN
    if direction == "left_to_right":
        source_valid &= source_disparity > 0
    else:
        source_valid &= source_disparity < 0
        source_valid &= source_disparity > -num_disparities

    ys, xs = np.nonzero(source_valid)
    stats = {
        "source_semantic_pixels": int(len(xs)),
        "valid_disparity_pixels": 0,
        "cycle_rejected_pixels": 0,
        "out_of_bounds_pixels": 0,
        "candidate_pixels": 0,
        "support_rejected_pixels": 0,
        "conflict_pixels": 0,
    }
    if len(xs):
        disparities = source_disparity[ys, xs].astype(np.float32)
        # OpenCV disparity is x_source - x_target in both matcher directions.
        target_x = np.rint(xs.astype(np.float32) - disparities).astype(np.int32)
        in_bounds = (target_x >= 0) & (target_x < width)
        stats["out_of_bounds_pixels"] = int(np.count_nonzero(~in_bounds))
        valid_idx = np.nonzero(in_bounds)[0]
        if len(valid_idx):
            valid_y = ys[valid_idx]
            valid_x = xs[valid_idx]
            valid_target_x = target_x[valid_idx]
            reverse_values = reverse_disparity[valid_y, valid_target_x]
            reverse_ok = np.isfinite(reverse_values)
            if direction == "left_to_right":
                reverse_ok &= reverse_values < 0
                reverse_ok &= reverse_values > -num_disparities
            else:
                reverse_ok &= reverse_values > 0
            cycle_ok = reverse_ok & (np.abs(disparities[valid_idx] + reverse_values) <= cycle_threshold)
            stats["valid_disparity_pixels"] = int(np.count_nonzero(reverse_ok))
            stats["cycle_rejected_pixels"] = int(np.count_nonzero(reverse_ok & ~cycle_ok) + np.count_nonzero(~reverse_ok))
            if np.any(cycle_ok):
                cy = valid_y[cycle_ok]
                cx = valid_target_x[cycle_ok]
                cl = source_labels[valid_y[cycle_ok], valid_x[cycle_ok]].astype(np.intp)
                np.add.at(votes, (cy, cx, cl), 1)

    kernel = (support_window, support_window)
    tree_votes = cv2.boxFilter(votes[:, :, LABEL_TREE], cv2.CV_32S, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT)
    road_votes = cv2.boxFilter(votes[:, :, LABEL_ROAD], cv2.CV_32S, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT)
    total = tree_votes + road_votes
    best_is_tree = tree_votes >= road_votes
    best_count = np.where(best_is_tree, tree_votes, road_votes)
    has_votes = total > 0
    conflict = (tree_votes > 0) & (road_votes > 0)
    accepted = has_votes & (best_count >= min_support) & (best_count.astype(np.float32) / np.maximum(total, 1) >= dominance_ratio)
    candidates[accepted & best_is_tree] = LABEL_TREE
    candidates[accepted & ~best_is_tree] = LABEL_ROAD
    stats["conflict_pixels"] = int(np.count_nonzero(conflict))
    stats["support_rejected_pixels"] = int(np.count_nonzero(has_votes & ~accepted))
    stats["candidate_pixels"] = int(np.count_nonzero(accepted))
    return candidates, stats, accepted


def fuse_labels(base: np.ndarray, projected: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    if base.shape != projected.shape:
        raise ValueError("base and projected labels must have identical shapes")
    fused = base.astype(np.uint8, copy=True)
    unknown = base == LABEL_UNKNOWN
    tree = projected == LABEL_TREE
    road = projected == LABEL_ROAD
    fused[unknown & tree] = LABEL_TREE
    fused[unknown & road] = LABEL_ROAD
    return fused, {
        "unknown_to_tree": int(np.count_nonzero(unknown & tree)),
        "unknown_to_road": int(np.count_nonzero(unknown & road)),
        "base_nonunknown_preserved": int(np.count_nonzero(~unknown & (projected != LABEL_UNKNOWN))),
    }


def _mask_from_result(result, shape: Tuple[int, int]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.uint8)
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return labels
    data = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
    road = np.zeros(shape, dtype=bool)
    tree = np.zeros(shape, dtype=bool)
    for mask, cls in zip(data, classes):
        resized = _resize_mask(mask > 0.5, shape)
        if cls == 0:
            road |= resized > 0
        elif cls == 1:
            tree |= resized > 0
    # Tree wins overlap, matching the historical mask definition.
    labels[road] = LABEL_ROAD
    labels[tree] = LABEL_TREE
    if np.any(tree):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tree_dilated = cv2.dilate(tree.astype(np.uint8), kernel, iterations=1) > 0
        labels[tree_dilated] = LABEL_TREE
    return labels


def _colorize(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    colors[labels == LABEL_TREE] = (0, 0, 255)
    colors[labels == LABEL_ROAD] = (0, 200, 0)
    return colors


def _overlay(gray: np.ndarray, labels: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = _colorize(labels)
    return cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0.0)


def _disparity_view(disparity: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disparity)
    view = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(disparity[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        view[valid] = np.clip((disparity[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(view, cv2.COLORMAP_TURBO)


def _cycle_errors(source: np.ndarray, reverse: np.ndarray, direction: str, num_disparities: int = 256) -> np.ndarray:
    if direction == "left_to_right":
        valid = np.isfinite(source) & (source > 0)
    else:
        valid = np.isfinite(source) & (source < 0)
        valid &= source > -num_disparities
    tx = np.rint(np.indices(source.shape)[1] - source).astype(np.int32)
    valid &= tx >= 0
    valid &= tx < source.shape[1]
    ys, xs = np.nonzero(valid)
    if not len(xs):
        return np.array([], dtype=np.float32)
    reverse_values = reverse[ys, tx[ys, xs]]
    valid_reverse = np.isfinite(reverse_values)
    if direction == "left_to_right":
        valid_reverse &= reverse_values < 0
        valid_reverse &= reverse_values > -num_disparities
    else:
        valid_reverse &= reverse_values > 0
    return np.abs(source[ys[valid_reverse], xs[valid_reverse]] + reverse_values[valid_reverse]).astype(np.float32)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8)):
        raise IOError(f"failed to write {path}")


def _write_disparity(path: Path, disparity: np.ndarray, offset: int = 32768) -> None:
    """Store signed disparity at 1/16 px precision in a uint16 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.zeros(disparity.shape, dtype=np.uint16)
    valid = np.isfinite(disparity)
    values = np.rint(disparity[valid] * 16.0).astype(np.int32) + offset
    values = np.clip(values, 1, 65535)
    encoded[valid] = values.astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise IOError(f"failed to write {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_indices(left_dir: Path, right_dir: Path) -> List[int]:
    left = {int(p.stem) for p in left_dir.glob("*.png") if p.stem.isdigit()}
    right = {int(p.stem) for p in right_dir.glob("*.png") if p.stem.isdigit()}
    common = sorted(left & right)
    if not common:
        raise RuntimeError("no paired frames found")
    if left != right:
        missing_left = sorted(right - left)[:10]
        missing_right = sorted(left - right)[:10]
        raise RuntimeError(f"left/right frame sets differ: missing_left={missing_left}, missing_right={missing_right}")
    return common


def _make_visualization(
    left: np.ndarray,
    right: np.ndarray,
    left_fused: np.ndarray,
    right_fused: np.ndarray,
    left_disparity: np.ndarray,
    right_disparity: np.ndarray,
    output_scale: float,
) -> np.ndarray:
    panels = [
        cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
        _overlay(left, left_fused),
        _disparity_view(left_disparity),
        cv2.cvtColor(right, cv2.COLOR_GRAY2BGR),
        _overlay(right, right_fused),
        _disparity_view(right_disparity),
    ]
    target_w = max(1, int(round(left.shape[1] * output_scale)))
    target_h = max(1, int(round(left.shape[0] * output_scale)))
    panels = [cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA) for panel in panels]
    return np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])


def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    output = Path(args.output)
    left_dir = dataset / "images_cam2_sr22555667"
    right_dir = dataset / "images_cam3_sr22555660"
    model_path = Path(args.model)
    frames = _frame_indices(left_dir, right_dir)
    selected = [i for i in frames if args.start <= i < args.end]
    if not selected:
        raise RuntimeError("selected frame range is empty")

    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    work_scale = float(args.scale)
    if not (0 < work_scale <= 1):
        raise ValueError("scale must be in (0, 1]")
    sample = cv2.imread(str(left_dir / f"{selected[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if sample is None:
        raise RuntimeError("failed to read sample frame")
    height, width = sample.shape
    work_shape = (max(1, int(round(height * work_scale))), max(1, int(round(width * work_scale))))
    left_matcher = build_sgbm(work_scale, "left", args.num_disparities)
    right_matcher = build_sgbm(work_scale, "right", args.num_disparities)

    for sub in ("masks/cam2", "masks/cam3", "projected_masks/cam2", "projected_masks/cam3", "fused_masks/cam2", "fused_masks/cam3", "disparity/cam2_to_cam3", "disparity/cam3_to_cam2", "visualizations"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    csv_path = output / "summary.csv"
    fields = [
        "index", "left_valid_percent", "right_valid_percent", "left_projected_pixels", "right_projected_pixels",
        "left_fused_pixels", "right_fused_pixels", "left_conflict_pixels", "right_conflict_pixels",
        "cycle_left_median_px", "cycle_right_median_px", "yolo_time_ms", "sgbm_time_ms", "total_time_ms",
    ]
    rows = []
    total_start = time.perf_counter()
    batch_size = max(1, int(args.batch_size))
    with ThreadPoolExecutor(max_workers=2) as stereo_pool:
      for batch_start in tqdm(range(0, len(selected), batch_size), desc="W01 SGBM"):
        batch_ids = selected[batch_start:batch_start + batch_size]
        left_images = [cv2.imread(str(left_dir / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) for i in batch_ids]
        right_images = [cv2.imread(str(right_dir / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) for i in batch_ids]
        if any(im is None for im in left_images + right_images):
            raise RuntimeError(f"failed to decode batch beginning at {batch_ids[0]}")
        yolo_start = time.perf_counter()
        left_results = model.predict(left_images, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det, retina_masks=True, device=device, verbose=False)
        right_results = model.predict(right_images, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det, retina_masks=True, device=device, verbose=False)
        yolo_ms = (time.perf_counter() - yolo_start) * 1000.0
        for frame_id, left, right, left_result, right_result in zip(batch_ids, left_images, right_images, left_results, right_results):
            frame_start = time.perf_counter()
            left_mask = _mask_from_result(left_result, left.shape)
            right_mask = _mask_from_result(right_result, right.shape)
            left_work = cv2.resize(left, (work_shape[1], work_shape[0]), interpolation=cv2.INTER_AREA)
            right_work = cv2.resize(right, (work_shape[1], work_shape[0]), interpolation=cv2.INTER_AREA)
            left_mask_work = _resize_mask(left_mask, work_shape)
            right_mask_work = _resize_mask(right_mask, work_shape)
            sgbm_start = time.perf_counter()
            left_future = stereo_pool.submit(left_matcher.compute, left_work, right_work)
            right_future = stereo_pool.submit(right_matcher.compute, right_work, left_work)
            disp_left = left_future.result().astype(np.float32) / 16.0
            disp_right = right_future.result().astype(np.float32) / 16.0
            to_right_work, right_stats, _ = project_labels(left_mask_work, disp_left, disp_right, "left_to_right", args.lr_threshold, args.support_window, args.min_support, args.dominance_ratio, args.num_disparities)
            to_left_work, left_stats, _ = project_labels(right_mask_work, disp_right, disp_left, "right_to_left", args.lr_threshold, args.support_window, args.min_support, args.dominance_ratio, args.num_disparities)
            sgbm_ms = (time.perf_counter() - sgbm_start) * 1000.0
            left_projected = _resize_mask(to_left_work, left.shape)
            right_projected = _resize_mask(to_right_work, right.shape)
            left_fused, left_fuse_stats = fuse_labels(left_mask, left_projected)
            right_fused, right_fuse_stats = fuse_labels(right_mask, right_projected)
            stem = f"{frame_id:06d}"
            _write_mask(output / "masks/cam2" / f"{stem}.png", left_mask)
            _write_mask(output / "masks/cam3" / f"{stem}.png", right_mask)
            _write_mask(output / "projected_masks/cam2" / f"{stem}.png", left_projected)
            _write_mask(output / "projected_masks/cam3" / f"{stem}.png", right_projected)
            _write_mask(output / "fused_masks/cam2" / f"{stem}.png", left_fused)
            _write_mask(output / "fused_masks/cam3" / f"{stem}.png", right_fused)
            _write_disparity(output / "disparity/cam2_to_cam3" / f"{stem}.png", disp_left)
            _write_disparity(output / "disparity/cam3_to_cam2" / f"{stem}.png", disp_right)
            save_visualization = frame_id < args.visualization_first or (
                args.visualization_every > 0 and frame_id % args.visualization_every == 0
            )
            if save_visualization:
                visualization = _make_visualization(
                    left, right, left_fused, right_fused, disp_left, disp_right, args.visualization_scale
                )
                _write_mask(output / "visualizations" / f"frame_{stem}.png", visualization)
                _write_mask(output / "visualizations" / f"disparity_cam2_to_cam3_{stem}.png", _disparity_view(disp_left))
                _write_mask(output / "visualizations" / f"disparity_cam3_to_cam2_{stem}.png", _disparity_view(disp_right))
            valid_left = np.isfinite(disp_left) & (disp_left > 0)
            valid_right = np.isfinite(disp_right) & (disp_right < 0) & (disp_right > -args.num_disparities)
            cycle_left = _cycle_errors(disp_left, disp_right, "left_to_right", args.num_disparities)
            cycle_right = _cycle_errors(disp_right, disp_left, "right_to_left", args.num_disparities)
            rows.append({
                "index": frame_id,
                "left_valid_percent": 100.0 * float(np.count_nonzero(valid_left)) / valid_left.size,
                "right_valid_percent": 100.0 * float(np.count_nonzero(valid_right)) / valid_right.size,
                "left_projected_pixels": int(np.count_nonzero(left_projected)),
                "right_projected_pixels": int(np.count_nonzero(right_projected)),
                "left_fused_pixels": int(np.count_nonzero(left_fused)),
                "right_fused_pixels": int(np.count_nonzero(right_fused)),
                "left_conflict_pixels": left_stats["conflict_pixels"],
                "right_conflict_pixels": right_stats["conflict_pixels"],
                "cycle_left_median_px": float(np.median(cycle_left)) if cycle_left.size else float("nan"),
                "cycle_right_median_px": float(np.median(cycle_right)) if cycle_right.size else float("nan"),
                "yolo_time_ms": yolo_ms / len(batch_ids),
                "sgbm_time_ms": sgbm_ms,
                "total_time_ms": (time.perf_counter() - frame_start) * 1000.0,
            })
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": str(dataset.resolve()),
        "model": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "model_names": {"0": "road", "1": "tree"},
        "frames_total": len(selected),
        "frame_start": selected[0],
        "frame_end_exclusive": selected[-1] + 1,
        "image_shape": [height, width],
        "scale": work_scale,
        "sgbm": {"num_disparities": args.num_disparities, "block_size": 5, "mode": "SGBM_3WAY"},
        "disparity_storage": {"format": "uint16_png", "scale": 16, "offset": 32768, "invalid": 0},
        "projection": {"cycle_threshold_px": args.lr_threshold, "support_window": args.support_window, "min_support": args.min_support, "dominance_ratio": args.dominance_ratio, "fusion": "preserve_target_nonunknown"},
        "visualizations": {"first": args.visualization_first, "every": args.visualization_every, "scale": args.visualization_scale},
        "elapsed_seconds": time.perf_counter() - total_start,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/W01_13Hz")
    parser.add_argument("--model", default="YOLO/yolo_workspace/train_my/runs/segment/yolov8s-seg-base/weights/best.pt")
    parser.add_argument("--output", default="results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=9210)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--num-disparities", type=int, default=256)
    parser.add_argument("--lr-threshold", type=float, default=2.0)
    parser.add_argument("--support-window", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--dominance-ratio", type=float, default=0.6)
    parser.add_argument("--visualization-first", type=int, default=10)
    parser.add_argument("--visualization-every", type=int, default=100)
    parser.add_argument("--visualization-scale", type=float, default=0.35)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
