# W01 Online Turn Fallback

## Purpose

The W01 semantic tracker originally filtered ORB features using the fused
static semantic mask. In the turn/U-turn interval around frames `7213-7250`,
the mask temporarily removed most useful features even though the original
all-feature ORB-SLAM3 tracker remained healthy. The resulting semantic run
entered `RECENTLY_LOST` and could create a second map.

This change keeps semantic filtering and geometric promotion during normal
motion, but temporarily restores all ORB features for tracking when semantic
support or the previous tracking result becomes critically weak.

## Algorithm Changes

The existing semantic fallback and geometric promotion remain enabled. The
new online fallback adds the following behavior:

1. Extract and retain the raw left/right ORB feature sets before semantic
   selection.
2. Starting at frame `1000`, trigger full-feature fallback if either camera
   has fewer than `50` static features, either camera has grid coverage below
   `0.10`, or the previous frame has fewer than `50` tracking inliers.
3. Keep the full-feature mode for up to `24` consecutive frames after a
   trigger. The non-IMU stereo path passes `mLastFrame` so this hold state is
   actually available to the next frame.
4. Use all raw ORB features to track existing map points during fallback.
5. Do not create new MapPoints or run semantic/geometric promotion while
   `mbTurnFallback` is set. This prevents uncertain fallback features from
   immediately contaminating the map.
6. Return to normal semantic selection after the hold window, unless another
   trigger is observed.

The semantic-quality thresholds are absolute current-frame thresholds; they
are not a frame-to-frame derivative. The older ordinary fallback remains
available with its less strict `300` static-feature and `0.15` grid-coverage
thresholds, but it only adds the configured limited fallback feature budget.
The current detector does not yet use an explicit rotation velocity, optical
flow, or motion-direction estimate.

## Configuration

Use [`configs/finnforest_w01_stereo_rectified_turn_online.yaml`](../configs/finnforest_w01_stereo_rectified_turn_online.yaml):

```yaml
Semantic.EnableTurnFallback: 1
Semantic.TurnFallbackOnline: 1
Semantic.TurnFallbackHoldFrames: 24
Semantic.TurnFallbackMinFrame: 1000
Semantic.TurnFallbackMinStaticFeatures: 50
Semantic.TurnFallbackMinGridCoverage: 0.10
Semantic.TurnFallbackStartFrame: -1
Semantic.TurnFallbackEndFrame: -1
Semantic.RecoveryTimeoutSec: 3.0
```

The fixed-window configuration is intended only for diagnostic upper-bound
experiments: `configs/finnforest_w01_stereo_rectified_turn_window.yaml`.

## Evaluation Results

The full W01 sequence contains `9210` frames. Results below are from the
local experiment outputs; generated results are ignored by Git, while this
document records the reproducible summary.

| Run | Status | Maps | Turn window `7213-7250` | Low-tree window `2783-2788` |
|---|---|---:|---|---|
| `run_01` | single-map success, pre-hold fix | 1 | `38/38 OK`, mean `193.1`, min `116` inliers | all `OK`, mean `149.8` |
| `run_02` | failed, switch at `7253` | 2 | `0/38 OK`, mean `0.5` | all `OK`, mean `159.2` |
| `run_03` | failed, old initialization gate | 2, switch at `306` | `38/38 OK`, mean `212.1` | all `OK`, mean `161.8` |
| `run_04` | failed, old stereo previous-frame path | 2, switch at `1056` | `38/38 OK`, mean `205.7` | all `OK`, mean `159.0` |
| `run_05` | single-map success, hold fix | 1 | `38/38 OK`, mean `268.1`, min `131` | all `OK`, mean `152.0` |
| `run_06` | incomplete | process segfault near frame `5812` | not evaluated | not evaluated |

For the successful single-map runs, global metrics are defined because the
trajectory is in one coordinate system:

| Run | Translation APE RMSE | Translation RPE RMSE, 100 frames |
|---|---:|---:|
| `run_01` | `3.3298 m` | `1.2972 m` |
| `run_05` | `5.3004 m` | `1.3118 m` |

The APE difference reflects different fallback behavior and asynchronous map
optimization; the similar 100-frame RPE values indicate comparable local
motion consistency. Multi-map runs must report global APE/RPE as `N/A` unless
cross-map transforms are supplied.

## Reproduction

Build the semantic runner and run the full sequence:

```bash
cmake --build third_party/ORB_SLAM3/build --target finnforest_stereo_semantic -j2
taskset -c 0-7 third_party/ORB_SLAM3/Examples/Stereo/finnforest_stereo_semantic \
  third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  configs/finnforest_w01_stereo_rectified_turn_online.yaml \
  data/W01_13Hz \
  results/2026-09-04_w01_semantic_projection_sgbm_full_masks/fused_masks \
  data/W01_13Hz/timestampSecNanoSec_W01.txt \
  results/2026-09-28_w01_experiments/turn_fallback_online/full_9210/run_new 9210
```

Evaluate a single-map trajectory with:

```bash
python3 scripts/evaluate_w01_multimap.py \
  --trajectory-with-map results/.../trajectory_map_0_semantic.csv \
  --groundtruth data/W01_13Hz/GT_W01.txt \
  --timestamps data/W01_13Hz/timestampSecNanoSec_W01.txt \
  --output results/.../evaluation \
  --last-frame 9209 \
  --alignment se3
```

Verification for this change: `pytest -q` passes `10` tests and the semantic
runner builds successfully. The fallback heuristic should not yet be treated
as a final rotation-aware turn detector, and the `run_06` shutdown crash
requires separate debugger-based investigation.
