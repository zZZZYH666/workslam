import numpy as np

from scripts.semantic_projection_sgbm import fuse_labels, project_labels


def test_projection_accepts_consistent_tree_and_fills_unknown():
    labels = np.zeros((7, 10), dtype=np.uint8)
    labels[2:5, 2:6] = 1
    left = np.zeros_like(labels, dtype=np.float32)
    right = np.zeros_like(labels, dtype=np.float32)
    left[:, 2:8] = 2
    right[:, :8] = -2
    projected, stats, _ = project_labels(labels, left, right, "left_to_right", cycle_threshold=0.1, min_support=3)
    assert stats["cycle_rejected_pixels"] == 0
    assert np.count_nonzero(projected == 1) > 0


def test_projection_rejects_cycle_inconsistent_pixels():
    labels = np.ones((5, 8), dtype=np.uint8)
    left = np.full_like(labels, 2, dtype=np.float32)
    right = np.full_like(labels, -5, dtype=np.float32)
    projected, stats, _ = project_labels(labels, left, right, "left_to_right", cycle_threshold=0.1)
    assert np.count_nonzero(projected) == 0
    assert stats["cycle_rejected_pixels"] > 0


def test_right_to_left_projection_moves_pixels_right():
    labels = np.zeros((7, 12), dtype=np.uint8)
    labels[2:5, 2:6] = 2
    right = np.zeros_like(labels, dtype=np.float32)
    left = np.zeros_like(labels, dtype=np.float32)
    right[:, :8] = -2
    left[:, 2:10] = 2
    projected, stats, _ = project_labels(labels, right, left, "right_to_left", cycle_threshold=0.1, min_support=3)
    ys, xs = np.nonzero(projected == 2)
    assert stats["cycle_rejected_pixels"] == 0
    assert xs.min() >= 3
    assert xs.max() > 5


def test_fusion_preserves_existing_labels():
    base = np.array([[0, 1, 2, 0]], dtype=np.uint8)
    projected = np.array([[1, 2, 1, 0]], dtype=np.uint8)
    fused, stats = fuse_labels(base, projected)
    np.testing.assert_array_equal(fused, np.array([[1, 1, 2, 0]], dtype=np.uint8))
    assert stats["unknown_to_tree"] == 1
    assert stats["base_nonunknown_preserved"] == 2
