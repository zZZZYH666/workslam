# v2.1 Provisional APE 修复版

本版本针对 W01 前 7254 帧中 provisional 在 5000–6500 帧平移 APE 扩大的问题，修正了临时候选点的双目来源统计、观测去重和重投影误差记录，并增加按年龄和质量准入的闭环支持。

## 主要变更

- 左右目来源分别统计；一侧静态、一侧回退时保留静态证据。
- 可见帧、匹配帧和关键帧观测去重；无效重投影误差不再以 0 计入。
- 新增 `MapPoint::IsLoopEligible()`，允许达到年龄/质量门槛的 provisional 支持闭环候选，但融合和 Global BA 仍过滤 provisional。
- 新增 `semantic_lifecycle_events.csv` 和 `backend_interval_stats.csv`。
- 新增 `semantic_mappoint_lifecycle_test`。

## 默认配置

`Semantic.EnableProvisionalLoopSupport: 1`，准入门槛为可见帧 5、关键帧 2、匹配率 0.60、最近观测不超过 8 个关键帧、动态观测为 0；闭环可信内点至少 15，provisional 占比不超过 30%。

## 验证

- `semantic_fallback_unit_test`：通过。
- `semantic_mappoint_lifecycle_test`：通过。
- `tests/test_semantic_projection_sgbm.py`：4 项通过。
- 200 帧 smoke：单地图完成，生成新增 CSV。

本版本不包含时序补偿、PoseOptimization 语义加权或 BA 连续语义权重。
