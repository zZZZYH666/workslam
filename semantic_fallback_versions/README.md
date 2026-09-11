# Semantic Fallback 版本归档

本目录集中保存当前实验的回退逻辑代码及后续完善版本。

## 目录约定

- 每次修改新建一个独立子目录；
- 子目录中保存当次代码副本、说明文档及必要的配置/测试文件；
- 不直接覆盖已有版本，保证实验结果可追溯和可复现。

版本索引：

- `semantic_fallback_archive/`：原始回退实现归档。
- `v2_provisional_mappoint/`：临时候选 MapPoint 的晋升/淘汰与后端过滤。
- `v2_1_provisional_ape_fix_2026-09-05/`：双目来源、观测去重、重投影记录及 provisional 闭环准入修复。
- `v2_2_mixed_source_local_ba_candidate_2026-09-07/`：mixed 双目来源按关键帧聚合，并在 Local BA 中将 provisional 信息矩阵降为 0.25；4200 帧检查通过，但 9210 帧仍发生地图重建，仅作为诊断候选版。
- `v3_soft_selection/`：仅候选排序阶段的软语义权重（需在 v2 验收后启用）。

完整 9210 帧重复实验使用 `scripts/run_w01_repeated_experiments.py`，聚合使用
`scripts/aggregate_w01_repeated_experiments.py`。脚本不会替用户猜测数据集路径，
执行前请用 `--dry-run` 检查命令和配置。
