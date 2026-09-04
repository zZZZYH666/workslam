# v2 provisional MapPoint

本版本在 W01 静态优先回退的基础上加入 MapPoint 生命周期管理。回退来源点创建为
`PROVISIONAL`，在 `LocalMapping::MapPointCulling()` 中按可见帧、关键帧观测、
匹配率、重投影误差和静态来源比例进行晋升或淘汰。静态来源点保持原有流程。

## 范围

- 当前帧跟踪和短期匹配仍可使用 provisional 点。
- Local/Global BA、重定位及回环路径过滤 provisional 点。
- 本版没有时序掩码补偿，也没有连续语义信息矩阵。
- 输入为二值掩码；静态来源比例是语义稳定性的代理。

## 验证

关闭 `Semantic.EnableMapPointPromotion` 可复现硬回退行为。纯逻辑测试位于
`tests/`；完整序列用仓库的重复实验脚本执行，并保存到新的结果目录。

`source/` 已包含当前修复后的核心头文件、实现文件、Stereo 实验入口及 CMakeLists，
可与工作区对应文件逐项按 SHA-256 核对。
