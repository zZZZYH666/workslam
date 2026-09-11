#!/usr/bin/env python3
"""Create old-preview-style random semantic mask comparison panels."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 1] = (0, 180, 0)      # tree
    out[mask == 2] = (190, 125, 35)   # road
    return out


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1.0 - alpha, colorize(mask), alpha, 0.0)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    bar_h = 32
    canvas = np.zeros((image.shape[0] + bar_h, image.shape[1], 3), dtype=np.uint8)
    canvas[bar_h:] = image
    cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04")
    p.add_argument("--dataset", default="data/W01_13Hz")
    p.add_argument("--output", default="results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04/sample_frames")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--scale", type=float, default=0.5)
    args = p.parse_args()

    root = Path(args.input)
    dataset = Path(args.dataset)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    left_dir = dataset / "images_cam2_sr22555667"
    right_dir = dataset / "images_cam3_sr22555660"
    available = sorted(int(x.stem) for x in (root / "masks/cam2").glob("*.png") if x.stem.isdigit())
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(available, min(args.count, len(available))))
    with (out / "sampled_indices.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["sample_order", "frame_index", "seed"])
        writer.writeheader()
        for order, idx in enumerate(selected):
            writer.writerow({"sample_order": order, "frame_index": idx, "seed": args.seed})

    for order, idx in enumerate(selected):
        stem = f"{idx:06d}"
        left = cv2.imread(str(left_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        left_mask = cv2.imread(str(root / "masks/cam2" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        right_mask = cv2.imread(str(root / "masks/cam3" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        left_fused = cv2.imread(str(root / "fused_masks/cam2" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        right_fused = cv2.imread(str(root / "fused_masks/cam3" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if any(x is None for x in (left, right, left_mask, right_mask, left_fused, right_fused)):
            raise RuntimeError(f"missing input for frame {idx}")
        panels = [
            add_title(overlay(left, left_mask), "cam2 mask"),
            add_title(overlay(right, right_mask), "cam3 mask"),
            add_title(overlay(left, left_fused), "cam2 fused mask"),
            add_title(overlay(right, right_fused), "cam3 fused mask"),
        ]
        target_w = int(round(left.shape[1] * args.scale))
        target_h = int(round((left.shape[0] + 32) * args.scale))
        panels = [cv2.resize(x, (target_w, target_h), interpolation=cv2.INTER_AREA) for x in panels]
        canvas = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        if not cv2.imwrite(str(out / f"{order:02d}_frame_{stem}.png"), canvas):
            raise RuntimeError(f"failed to write frame {idx}")


if __name__ == "__main__":
    main()
