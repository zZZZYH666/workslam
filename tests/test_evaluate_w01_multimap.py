from pathlib import Path

import numpy as np

from scripts.evaluate_w01_multimap import (
    rigid_alignment,
    read_trajectory_with_map_files,
    resolve_map_transforms,
    same_map_rpe,
    sim3_alignment,
)


def test_single_map_can_use_its_local_frame_as_global_frame():
    frames = [0, 1, 2]
    map_ids = {frame: "0" for frame in frames}
    transforms, source = resolve_map_transforms(True, frames, map_ids, None)
    assert source == "identity_single_map"
    np.testing.assert_allclose(transforms["0"], np.eye(4))


def test_multi_map_still_requires_explicit_transforms():
    frames = [0, 1, 2]
    map_ids = {0: "0", 1: "0", 2: "1"}
    transforms, source = resolve_map_transforms(True, frames, map_ids, None)
    assert transforms is None
    assert source is None


def make_poses(points):
    poses = np.repeat(np.eye(4)[None, :, :], len(points), axis=0)
    poses[:, :3, 3] = np.asarray(points)
    return poses


def test_rigid_alignment_recovers_transform():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    target = source + np.array([3, -2, 5])
    transform = rigid_alignment(source, target)
    np.testing.assert_allclose((transform[:3, :3] @ source.T).T + transform[:3, 3], target, atol=1e-12)


def test_sim3_alignment_recovers_scale():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    target = 2.5 * source + np.array([3, -2, 5])
    transform = sim3_alignment(source, target)
    np.testing.assert_allclose((transform[:3, :3] @ source.T).T + transform[:3, 3], target, atol=1e-12)


def test_same_map_rpe_excludes_cross_map_pair():
    frames = [0, 1, 2, 3]
    poses = make_poses([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    truth = make_poses([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    rows = same_map_rpe(frames, poses, truth, {0: "a", 1: "a", 2: "b", 3: "b"}, deltas=(1,))
    assert rows[0]["samples"] == 2
    assert rows[0]["scope"] == "same_map"


def test_map_split_reader_accepts_map_id_at_end(tmp_path):
    path_a = Path(tmp_path) / "trajectory_map_0_baseline.csv"
    path_b = Path(tmp_path) / "trajectory_map_1_baseline.csv"
    header = "frame,timestamp_ns,tracking_state,pose_valid,tx,ty,tz,qx,qy,qz,qw,map_id\n"
    path_a.write_text(header + "0,100,2,1,0,0,0,0,0,0,1,0\n")
    path_b.write_text(header + "1,200,2,1,1,0,0,0,0,0,1,1\n")
    entries = read_trajectory_with_map_files([path_a, path_b], last_frame=1)
    assert [(frame, map_id) for frame, _pose, map_id in entries] == [(0, "0"), (1, "1")]
