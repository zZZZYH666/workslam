#!/usr/bin/env python3
"""Generate per-camera W01 hard semantic masks from a YOLO segmentation model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


LABEL_UNKNOWN = 0
LABEL_TREE = 1
LABEL_ROAD = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_indices(left_dir: Path, right_dir: Path) -> list[int]:
    left = {int(p.stem) for p in left_dir.glob("*.png") if p.stem.isdigit()}
    right = {int(p.stem) for p in right_dir.glob("*.png") if p.stem.isdigit()}
    return sorted(left & right)


def mask_from_result(result, shape: tuple[int, int], class_ids: dict[str, int]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.uint8)
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return labels
    data = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
    road_id = class_ids.get("road")
    tree_id = class_ids.get("tree")
    road = np.zeros(shape, dtype=bool)
    tree = np.zeros(shape, dtype=bool)
    for mask, cls in zip(data, classes):
        resized = cv2.resize((mask > 0.5).astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        if road_id is not None and int(cls) == road_id:
            road |= resized > 0
        elif tree_id is not None and int(cls) == tree_id:
            tree |= resized > 0
    labels[road] = LABEL_ROAD
    labels[tree] = LABEL_TREE
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/W01_13Hz")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--tree-dilate-radius", type=int, default=1)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    dataset = Path(args.dataset)
    model_path = Path(args.model)
    output = Path(args.output)
    left_dir = dataset / "images_cam2_sr22555667"
    right_dir = dataset / "images_cam3_sr22555660"
    frames = [i for i in frame_indices(left_dir, right_dir) if i >= args.start and (args.end is None or i < args.end)]
    if not frames:
        raise RuntimeError("no frames selected")
    if args.batch_size < 1 or args.tree_dilate_radius < 0:
        raise ValueError("batch-size must be positive and tree-dilate-radius nonnegative")

    model = YOLO(str(model_path))
    names = {str(k): str(v) for k, v in model.names.items()} if isinstance(model.names, dict) else {str(i): str(v) for i, v in enumerate(model.names)}
    normalized = {int(k): v.lower() for k, v in names.items()}
    class_ids = {name: cls for cls, name in normalized.items() if name in {"road", "tree"}}
    if "road" not in class_ids or "tree" not in class_ids:
        raise RuntimeError(f"model classes do not contain road/tree: {names}")
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    cam2_out = output / "masks_cam2"
    cam3_out = output / "masks_cam3"
    cam2_out.mkdir(parents=True, exist_ok=True)
    cam3_out.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    fields = ["index", "left_image", "right_image", "left_mask", "right_mask", "height", "width", "left_static_pixels", "right_static_pixels", "left_tree_pixels", "right_tree_pixels", "left_road_pixels", "right_road_pixels", "left_tree_instances", "right_tree_instances", "left_road_instances", "right_road_instances", "yolo_time_ms"]
    rows = []
    start_time = time.perf_counter()
    for batch_start in tqdm(range(0, len(frames), args.batch_size), desc="W01 hard masks"):
        batch_ids = frames[batch_start:batch_start + args.batch_size]
        left_images = [cv2.imread(str(left_dir / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) for i in batch_ids]
        right_images = [cv2.imread(str(right_dir / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) for i in batch_ids]
        if any(image is None for image in left_images + right_images):
            raise RuntimeError(f"failed to decode batch {batch_ids[0]}")
        infer_start = time.perf_counter()
        left_results = model.predict(left_images, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det, retina_masks=True, device=device, verbose=False)
        right_results = model.predict(right_images, conf=args.conf, imgsz=args.imgsz, max_det=args.max_det, retina_masks=True, device=device, verbose=False)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0 / len(batch_ids)
        for frame_id, left, right, left_result, right_result in zip(batch_ids, left_images, right_images, left_results, right_results):
            left_mask = mask_from_result(left_result, left.shape, class_ids)
            right_mask = mask_from_result(right_result, right.shape, class_ids)
            if args.tree_dilate_radius:
                size = 2 * args.tree_dilate_radius + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                left_tree = left_mask == LABEL_TREE
                right_tree = right_mask == LABEL_TREE
                left_mask[cv2.dilate(left_tree.astype(np.uint8), kernel) > 0] = LABEL_TREE
                right_mask[cv2.dilate(right_tree.astype(np.uint8), kernel) > 0] = LABEL_TREE
            stem = f"{frame_id:06d}.png"
            left_path = cam2_out / stem
            right_path = cam3_out / stem
            if not cv2.imwrite(str(left_path), left_mask) or not cv2.imwrite(str(right_path), right_mask):
                raise RuntimeError(f"failed to write mask for frame {stem}")
            rows.append({
                "index": frame_id,
                "left_image": str((left_dir / stem).resolve()),
                "right_image": str((right_dir / stem).resolve()),
                "left_mask": str(left_path.resolve()),
                "right_mask": str(right_path.resolve()),
                "height": left.shape[0], "width": left.shape[1],
                "left_static_pixels": int(np.count_nonzero(left_mask)), "right_static_pixels": int(np.count_nonzero(right_mask)),
                "left_tree_pixels": int(np.count_nonzero(left_mask == LABEL_TREE)), "right_tree_pixels": int(np.count_nonzero(right_mask == LABEL_TREE)),
                "left_road_pixels": int(np.count_nonzero(left_mask == LABEL_ROAD)), "right_road_pixels": int(np.count_nonzero(right_mask == LABEL_ROAD)),
                "left_tree_instances": int(sum(int(c) == class_ids["tree"] for c in left_result.boxes.cls.detach().cpu().numpy())) if left_result.boxes is not None else 0,
                "right_tree_instances": int(sum(int(c) == class_ids["tree"] for c in right_result.boxes.cls.detach().cpu().numpy())) if right_result.boxes is not None else 0,
                "left_road_instances": int(sum(int(c) == class_ids["road"] for c in left_result.boxes.cls.detach().cpu().numpy())) if left_result.boxes is not None else 0,
                "right_road_instances": int(sum(int(c) == class_ids["road"] for c in right_result.boxes.cls.detach().cpu().numpy())) if right_result.boxes is not None else 0,
                "yolo_time_ms": infer_ms,
            })
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": str(dataset.resolve()), "model": str(model_path.resolve()), "model_sha256": sha256(model_path),
        "model_names": names, "task": "segment", "confidence": args.conf, "imgsz": args.imgsz, "max_det": args.max_det,
        "batch_size": args.batch_size, "retina_masks": True, "device": device,
        "erode_radius_px": 0, "tree_dilate_radius_px": args.tree_dilate_radius,
        "label_mapping": {"0": "unknown", "1": "tree", "2": "road"}, "frame_count": len(rows),
        "frame_start": frames[0], "frame_end_exclusive": frames[-1] + 1,
        "image_shape": [int(rows[0]["height"]), int(rows[0]["width"])],
        "elapsed_seconds": time.perf_counter() - start_time,
        "totals": {"tree_pixels": sum(r["left_tree_pixels"] + r["right_tree_pixels"] for r in rows), "road_pixels": sum(r["left_road_pixels"] + r["right_road_pixels"] for r in rows), "frames_empty": sum(r["left_static_pixels"] == 0 and r["right_static_pixels"] == 0 for r in rows)},
    }
    (output / "mask_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
