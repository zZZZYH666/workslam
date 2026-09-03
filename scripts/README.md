# W01 语义掩码处理脚本说明

本目录包含当前 W01 双目语义掩码实验的处理和可视化脚本。脚本之间相互独立：主脚本负责生成结果，抽样脚本只读取已经生成的结果，不会重新运行 YOLO 或 SGBM。

## 目录内容

| 文件 | 用途 |
|---|---|
| [`semantic_projection_sgbm.py`](semantic_projection_sgbm.py) | YOLO 掩码生成、双向 StereoSGBM 视差、语义投影、投票筛选、融合和逐帧统计 |
| [`sample_semantic_frames.py`](sample_semantic_frames.py) | 从全量结果中固定随机抽取若干帧，生成左右目掩码与融合掩码拼图 |
| [`__init__.py`](__init__.py) | 将 `scripts` 声明为 Python 包 |

对应的单元测试位于 [`../tests/test_semantic_projection_sgbm.py`](../tests/test_semantic_projection_sgbm.py)。

## 1. semantic_projection_sgbm.py

### 功能

该脚本对 W01 双目序列逐帧执行以下处理：

1. 读取并配对 cam2/cam3 图像。
2. 使用 YOLOv8-Seg 分别推理左右图像。
3. 将模型类别转换为三值标签：`0=unknown`、`1=tree`、`2=road`。
4. 保留道路区域，对树木区域使用 `3x3` 椭圆核膨胀 1 像素，不做腐蚀。
5. 在 `scale=0.5` 下计算左右两个方向的 StereoSGBM 视差。
6. 使用正视差、目标坐标范围和左右 cycle error 过滤投影源像素。
7. 在目标像素 `3x3` 邻域中投票，要求至少 3 票且主标签比例不低于 0.6。
8. 只将通过筛选的 tree/road 标签写入目标 `unknown` 区域；已有标签不覆盖，tree/road 冲突保持 unknown。
9. 保存掩码、投影掩码、融合掩码、编码视差、可视化图和逐帧统计。

### 默认运行

在仓库根目录执行：

```bash
source infra/env.sh
python scripts/semantic_projection_sgbm.py
```

默认输入、模型和输出分别为：

```text
data/W01_13Hz
YOLO/yolo_workspace/train_my/runs/segment/yolov8s-seg-base/weights/best.pt
results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04
```

### 常用参数

```text
--start / --end              帧号范围，默认 0 到 9210
--batch-size                 YOLO 批大小，默认 4
--device                     推理设备，默认自动选择 CUDA 或 CPU
--conf                       YOLO 置信度阈值，默认 0.35
--imgsz                      YOLO 推理尺寸，默认 640
--max-det                    每张图最大检测数，默认 50
--scale                      SGBM 工作尺度，默认 0.5
--num-disparities            视差范围，默认 256，必须是 16 的倍数
--lr-threshold               左右 cycle error 阈值，默认 2.0 px
--support-window             投票窗口边长，默认 3
--min-support                最小投票数，默认 3
--dominance-ratio            主标签最小比例，默认 0.6
--visualization-first        保存开头若干帧，默认 10
--visualization-every        每隔多少帧保存可视化，默认 100
```

例如只处理前 200 帧：

```bash
python scripts/semantic_projection_sgbm.py \
  --start 0 --end 200 \
  --output results/result/w01_semantic_projection_sgbm_preview_200
```

### 输出结构

```text
<output>/
├── masks/cam2/                 左目 YOLO 基础掩码
├── masks/cam3/                 右目 YOLO 基础掩码
├── projected_masks/cam2/       投影到左目的候选掩码
├── projected_masks/cam3/       投影到右目的候选掩码
├── fused_masks/cam2/           左目最终融合掩码
├── fused_masks/cam3/           右目最终融合掩码
├── disparity/cam2_to_cam3/     左到右带符号视差
├── disparity/cam3_to_cam2/     右到左带符号视差
├── visualizations/             掩码叠加图和伪彩色视差图
├── summary.csv                 逐帧统计
├── config.json                 命令行参数
└── metadata.json               数据、模型哈希和运行元数据
```

掩码为 `uint8` PNG，像素值仅为 `0/1/2`。视差为 `uint16` PNG，编码规则为：

```text
disparity = (stored_value - 32768) / 16
stored_value = 0 表示无效视差
```

当前版本不生成或保留视频；需要视频时应使用单独的视频生成脚本或后处理流程。

## 2. sample_semantic_frames.py

### 功能

该脚本从主脚本已经生成的 `masks` 和 `fused_masks` 中随机抽样，将原始灰度图与彩色掩码叠加成 2x2 展示图：

```text
左上：cam2 mask
右上：cam3 mask
左下：cam2 fused mask
右下：cam3 fused mask
```

树木使用绿色显示，道路使用棕黄色显示，unknown 保留灰度背景。

### 默认运行

```bash
source infra/env.sh
python scripts/sample_semantic_frames.py
```

默认从 `w01_semantic_projection_sgbm_full_masks_2026-09-04` 抽取 20 帧，输出到：

```text
results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04/sample_frames
```

随机种子默认为 `20260831`，因此同一输入目录下可以复现相同的抽样帧。调整抽样数量或随机种子：

```bash
python scripts/sample_semantic_frames.py \
  --count 20 \
  --seed 20260831 \
  --input results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04 \
  --output results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04/sample_frames
```

脚本同时生成 `sampled_indices.csv`，记录抽样顺序、原始帧号和随机种子。

## 3. 测试

只测试投影和融合核心逻辑，不需要加载 YOLO 权重或读取 W01 全量数据：

```bash
source infra/env.sh
pytest -q tests/test_semantic_projection_sgbm.py
```

测试覆盖：

- 左到右的一致视差投影；
- cycle error 不一致时拒绝投影；
- 右到左投影方向；
- 融合时保留目标已有标签。

## 4. 当前正式实验

当前完整实验的说明见：

- [`../results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04/实验总结.md`](../results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04/实验总结.md)
- [`../results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04/report.md`](../results/result/w01_semantic_projection_sgbm_full_masks_2026-09-04/report.md)

正式实验使用 9210 帧 W01 数据，模型和参数以输出目录中的 `metadata.json`、`config.json` 为准。
