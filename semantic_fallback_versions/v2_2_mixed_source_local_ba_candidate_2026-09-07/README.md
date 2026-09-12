# v2.2 Mixed-source + Local BA 候选版

本版本从 `v2_1_provisional_ape_fix_2026-09-05` 派生，记录 2026-09-06
对 provisional 地图失稳问题进行逐步测试后保留下来的 B+C 变更。该版本是
**诊断候选版**，不是完整 9210 帧稳定性验收通过版。

## 代码变更

1. **双目 mixed 来源按关键帧聚合**
   - 同一关键帧的左右目不再作为两个独立语义样本累计。
   - 若一眼为静态来源、另一眼为 fallback 来源，则该关键帧保留静态证据，
     `SemanticStaticRatio()` 计为静态支持。
   - 对应实现：`source/MapPoint.cc`。

2. **Local BA 对 provisional 观测软降权**
   - provisional MapPoint 仍参加 Local BA，但单目、双目和右目边的信息矩阵统一乘
     `0.25`。
   - 已晋升或静态来源的 MapPoint 保持原权重 `1.0`。
   - 对应实现：`source/Optimizer.cc`。

3. **测试期望更新**
   - mixed stereo 观测的静态比例期望由 `0.5` 更新为 `1.0`。
   - 对应测试：`tests/test_semantic_mappoint_lifecycle.cc`。

## 配置

- 主配置：`config/finnforest_w01_stereo_rectified_mixed_local_ba.yaml`。
- provisional 年龄阈值保持 v2.1 的 `8`，未采用失败的 age-fix `20`。
- Local BA 的 `0.25` 当前是实现常量，不是 YAML 参数。
- 同时保留 v2.1 的 fallback、fallback-hard 和 provisional APE fix 配置用于对照。

## 验证记录

| 组合 | 范围 | 结果 |
|---|---:|---|
| B：mixed-source 聚合 | 4200 帧 | 单地图、全程 OK；通过短序列检查 |
| C：Local BA ×0.25 | 4200 帧 | 单地图、全程 OK；通过短序列检查 |
| B+C | 9210 帧 | 约 7206 帧地图重建；最终 2 张地图，不通过完整验收 |

B+C 完整运行统计为 9169 帧 `OK`、40 帧 `RECENTLY_LOST`、1 帧
`NO_IMAGES_YET`。完整逐步测试结论保存在 `validation/逐步修复测试报告.md`，原始
运行产物位于：

`results/result/2026-09-06_w01_provisional_improvement_repeat5/`

构建及逻辑测试结果：

- `finnforest_stereo_semantic`：构建通过；
- `semantic_fallback_unit_test`：通过；
- `semantic_mappoint_lifecycle_test`：通过。

## 未纳入的实验

“按最近语义关键帧观测计算 age，并将最大年龄改为 20”的 A 方案已回退。该方案在
4200 帧重复测试中约 2748 帧发生跟踪丢失和临时地图切换，因此不属于本候选版本。
其配置和失败产物仅保留在工作区实验记录中。

## 结论与使用限制

本版本证明 B、C 在 4200 帧范围内可运行，但没有解决 7100–7213 帧附近的完整序列
局部地图失稳。它适合复现实验和继续诊断，不应标记为稳定基线。
