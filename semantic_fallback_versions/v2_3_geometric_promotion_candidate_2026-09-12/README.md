# v2.3 Geometric Promotion 候选版

本版本从 `v2_2_mixed_source_local_ba_candidate_2026-09-07` 派生，为缺少静态语义
证据、但具有持续稳定几何观测的 provisional MapPoint 增加第二条晋升通道。本版本是
**精度改善候选版**，尚未通过完整 9210 帧单地图稳定性验收。

## 代码变更

1. **几何晋升通道**
   - 原语义晋升条件保持不变。
   - 开启 `Semantic.EnableGeometricPromotion` 后，unknown-only provisional 点可凭
     可见帧数、关键帧数、匹配率、平均重投影误差和动态观测数晋升。
   - 几何晋升仍要求 `Semantic.EnableMapPointPromotion=1`。

2. **可区分的生命周期事件**
   - 原有晋升记录改为 `promotion_semantic`。
   - 新通道记录为 `promotion_geometric`，便于从
     `semantic_lifecycle_events.csv` 独立统计。

3. **配置和测试**
   - 主配置：`config/finnforest_w01_stereo_rectified_geometric_promotion.yaml`。
   - 新增 unknown-only 点通过几何门槛、但不能通过语义门槛的生命周期测试。
   - 评估脚本：`evaluation/evaluate_w01_geometric_promotion.py`。

## 本次阈值

| 配置项 | 值 |
|---|---:|
| `GeometricPromotionMinVisibleFrames` | 8 |
| `GeometricPromotionMinKeyFrames` | 3 |
| `GeometricPromotionMinMatchRatio` | 0.70 |
| `GeometricPromotionMaxReprojectionError` | 1.5 px |
| `GeometricPromotionMaxDynamicObservations` | 0 |

## 构建与复现

在仓库根目录构建并执行逻辑测试：

```bash
cmake --build third_party/ORB_SLAM3/build -j2
ctest --test-dir third_party/ORB_SLAM3/build -R 'semantic_(fallback_unit|mappoint_lifecycle)_test' --output-on-failure
```

实验程序参数顺序为：

```text
finnforest_stereo_semantic vocabulary settings dataset fused_masks timestamps output_dir [max_frames] [start_index]
```

使用本目录配置运行 `0-7253` 共 7254 帧，再执行评估：

```bash
python3 scripts/evaluate_w01_geometric_promotion.py \
  --candidate results/result/2026-09-12_w01_geometric_promotion/prefix_7254/CameraTrajectory_semantic.txt \
  --output-dir results/result/2026-09-12_w01_geometric_promotion/prefix_7254/evaluation
```

评估依赖固定的 W01 真值、时间戳，以及脚本默认指向的 baseline 和 adaptive fallback
参考轨迹。完整运行命令中的数据集和 fused mask 路径应按本机实际位置传入。

## 实验结果

7254 帧独立前缀运行全程 `OK`、保持 1 张地图。三种方法在相同 7254 帧上分别进行
无尺度刚体 SE(3) 对齐后，APE RMSE 为：

| 方法 | Translation | Rotation |
|---|---:|---:|
| baseline | 3.6217 m | 8.4342 deg |
| adaptive fallback | 2.6998 m | 8.3116 deg |
| geometric promotion | **2.5902 m** | **8.3040 deg** |

相对 adaptive fallback，translation RMSE 降低 `4.1%`，rotation RMSE 降低
`0.09%`，后者整体上可视为持平。在重点区间 `5000-6700`，translation 从
`2.6213 m` 降至 `2.5477 m`，rotation 从 `9.1067 deg` 降至 `8.6523 deg`。

本次前缀运行记录 `10387` 次几何晋升和 `472` 次语义晋升。

## 稳定性限制

完整 9210 帧运行虽然处理完所有输入，但只有 `9128` 帧为 `OK`、`80` 帧为
`RECENTLY_LOST`，共出现 `82` 次局部地图跟踪失败，最终形成 2 张地图。保存的相机
轨迹只有 7252 行、7174 个唯一帧，不能用于声称完整 9210 帧 APE。

因此本版本仅证明几何晋升在一次连续 7254 帧运行中改善了平移精度，并改善了
`5000-6700` 区间的旋转精度；它不能替代稳定基线。完整产物和详细结论位于
`results/result/2026-09-12_w01_geometric_promotion/`。
