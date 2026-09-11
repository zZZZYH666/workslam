# v3 soft selection

本版本只在候选特征选择阶段使用连续权重，不修改 PoseOptimization、BA、回环或
MapPoint 观测信息矩阵。二值输入的默认映射为静态 `1.0`、Unknown 回退 `0.4`，
排序分数为 `ORB response × semantic weight`，并继续受网格覆盖和回退比例限制。

`Semantic.EnableSoftSelection: 0` 时应复现 v2 的硬筛选输出。该版本必须在
provisional 版本通过完整序列稳定性门槛后再进行重复实验。

`source/` 已包含当前修复后的核心实现、Stereo 实验入口及 CMakeLists；软选择仍由
配置项控制，关闭时保持硬筛选行为。
