# SemanticFallback 回退逻辑说明

## 1. 文件用途

`SemanticFallback.h` 汇总了当前 FinnForest W01 双目实验中的语义特征筛选与自适应回退逻辑。它位于独立归档目录中，便于实验复现、代码审查和后续迁移。

原始工程中的调用位置为 `third_party/ORB_SLAM3/src/Frame.cc`。该文件采用 header-only 实现，不需要额外的 `.cc` 文件或 CMake 源文件配置。

## 2. 核心数据结构

### `SelectionStats`

每次对单目图像进行筛选后返回的统计信息：

| 字段 | 含义 |
| --- | --- |
| `staticCount` | 掩膜静态区域内的原始特征数 |
| `selectedStaticCount` | 最终选中的静态区域特征数 |
| `fallbackCount` | 从未知区域补入的特征数 |
| `gridCoverage` | 静态特征占据的 8×6 网格比例 |

### `FallbackDecision`

表示是否应该触发回退，以及触发原因位：

| 字段 | 含义 |
| --- | --- |
| `requested` | 是否允许且需要启用回退 |
| `reason` | 回退原因的按位或结果 |

## 3. 特征选择流程

`SelectFeatures` 的处理步骤如下：

1. 根据语义掩膜将 ORB 特征分为静态区域和未知区域。掩膜为空时，所有特征视为静态特征。
2. 统计静态特征数量，并计算静态特征覆盖的 8×6 网格比例。
3. 在允许回退时，先使用网格轮询策略选择静态特征；否则保留所有静态特征。
4. 如果已请求回退且静态特征不足目标数量，则从未知区域补充特征。
5. 未知区域补充数量受 `targetFeatures × maxFallbackRatio` 限制。
6. 同步裁剪关键点和描述子，并通过 `sources` 标记来源：`0` 表示静态区域，`1` 表示回退特征。

## 4. 网格选择策略

`SelectByGrid` 使用 8 列 × 6 行网格：

- 每个网格单元内按 ORB response 从高到低排序；
- 按轮询顺序从各单元取第 1、2、3… 个特征；
- 直到达到请求数量或所有单元耗尽。

这种策略比单纯按 response 排序更能保持图像空间分布，减少特征集中在局部区域的问题。

## 5. 回退触发条件

`EvaluateRequest` 会检查左右目静态特征统计和上一帧跟踪质量。当满足任一条件且配置启用了回退（`Semantic.EnableFallback: 1`）时，触发回退：

- 左目或右目静态特征少于 `minStaticFeatures`；
- 左目或右目静态网格覆盖率低于 `minGridCoverage`；
- 上一帧跟踪内点数少于 `minTrackingInliers`。

首次筛选完成并执行双目匹配后，如果回退尚未触发但双目匹配数低于 `minStereoMatches`，`ShouldRetryForStereo` 会触发一次二次回退，并设置双目匹配不足原因位。

## 6. 原因位定义

`mnSemanticFallbackReason` 使用按位标记保存原因，可以同时表示多个原因：

| 位值 | 含义 |
| ---: | --- |
| `1` | 静态特征数量不足 |
| `2` | 静态网格覆盖率不足 |
| `4` | 上一帧跟踪质量不足 |
| `8` | 双目匹配数不足，触发二次回退 |

例如，原因值 `6`（`2 + 4`）表示网格覆盖率不足且上一帧跟踪质量不足。

## 7. 实验配置

当前回退配置位于 `configs/finnforest_w01_stereo_rectified_fallback.yaml`：

```yaml
Semantic.EnableFallback: 1
Semantic.TargetFeatures: 1000
Semantic.MinStaticFeatures: 300
Semantic.MinStereoMatches: 80
Semantic.MinTrackingInliers: 50
Semantic.MinGridCoverage: 0.15
Semantic.MaxFallbackRatio: 0.50
```

含义是：目标特征数为 1000，未知区域最多补充目标数量的 50%，即最多 500 个回退特征。

## 8. 调用关系

```text
Tracking::GrabImageStereo
        │
        ▼
Frame::Frame(..., SemanticConfig, previousTrackingInliers)
        │
        ├─ SemanticFallback::SelectFeatures（左右目初筛）
        ├─ SemanticFallback::EvaluateRequest（静态质量/历史跟踪判定）
        ├─ SemanticFallback::SelectFeatures（静态优先 + 未知区域补充）
        ├─ ComputeStereoMatches
        └─ SemanticFallback::ShouldRetryForStereo（必要时二次回退）
```

## 9. 归档文件

- `SemanticFallback.h`：当前实验实际使用的回退逻辑代码副本。
- `SemanticFallback.md`：本说明文档。

该目录是代码副本，不会自动替代原工程文件；对归档副本的修改不会影响 ORB-SLAM3 当前构建，反之亦然。
