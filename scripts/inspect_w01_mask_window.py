#!/usr/bin/env python3
"""Visualize W01 semantic masks around the map-switch failure window."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


LABEL_COLORS = {
    0: (25, 25, 25),       # unknown
    1: (40, 180, 40),      # tree
    2: (40, 90, 220),      # road
}


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        out[mask == label] = color
    return out


def labeled(image: np.ndarray, title: str, width: int = 620) -> np.ndarray:
    scale = width / image.shape[1]
    image = cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def make_frame(dataset: Path, mask_root: Path, frame: int, camera: str, output: Path) -> dict[str, int | float]:
    image_path = dataset / ("images_cam2_sr22555667" if camera == "cam2" else "images_cam3_sr22555660") / f"{frame:06d}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    raw = cv2.imread(str(mask_root / "masks" / camera / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    projected = cv2.imread(str(mask_root / "projected_masks" / camera / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    fused = cv2.imread(str(mask_root / "fused_masks" / camera / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if any(x is None for x in (image, raw, projected, fused)):
        raise RuntimeError(f"missing image or mask for {camera} frame {frame}")

    original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raw_color = colorize(raw)
    projected_color = colorize(projected)
    fused_color = colorize(fused)
    overlay = cv2.addWeighted(original, 0.62, fused_color, 0.38, 0.0)
    panels = [
        labeled(original, f"frame {frame} {camera}: image"),
        labeled(raw_color, "raw mask: green=tree, red=road"),
        labeled(projected_color, "reprojected mask"),
        labeled(overlay, "fused mask overlay"),
    ]
    montage = np.hstack(panels)
    cv2.imwrite(str(output / f"{camera}_{frame:06d}.png"), montage)

    row: dict[str, int | float] = {"frame": frame, "camera": camera}
    for name, mask in (("raw", raw), ("projected", projected), ("fused", fused)):
        for label in (1, 2):
            row[f"{name}_label_{label}_pixels"] = int(np.count_nonzero(mask == label))
        row[f"{name}_nonzero_pixels"] = int(np.count_nonzero(mask))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/W01_13Hz"))
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for camera in ("cam2", "cam3"):
        for frame in args.frames:
            rows.append(make_frame(args.dataset, args.mask_root, frame, camera, args.output))
    with (args.output / "mask_area_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
