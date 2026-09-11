# FinnForest：YOLOv8-Seg + ORB-SLAM3

当前工作区：`/root/workSLAM`。服务器刷新后，本工作区已于 2026-08-28 从空目录重新配置。

## 当前验收状态

| 组件 | 版本/状态 |
|---|---|
| Ubuntu | 22.04.4 LTS |
| GPU | NVIDIA GeForce RTX 5090，驱动 580.95.05 |
| Python | 3.10.12，虚拟环境 `.venv-finnforest` |
| PyTorch | 2.9.1+cu128，CUDA 可用 |
| torchvision | 0.24.1+cu128 |
| Ultralytics | 8.4.115 |
| Python OpenCV | 5.0.0 |
| C++ OpenCV | 4.5.4 |
| evo / rosbags | 1.37.0 / 0.11.3 |
| Pangolin | v0.6，已安装到 `.local` |
| ORB-SLAM3 | `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`，已编译 |

README 旧记录中的 PyTorch 2.11.0+cu128 与 torchvision 0.26.0+cu128 不在官方 CUDA 12.8 wheel 索引中。本次使用索引中实际可用并适配 RTX 5090 的 2.9.1/0.24.1 组合。

## 每次进入

```bash
cd /root/workSLAM
source infra/env.sh
./infra/verify.sh
```

`infra/env.sh` 默认使用虚拟环境中安装的 Ultralytics。只有确实要使用上传到 `YOLO/` 的本地源码时，才执行：

```bash
export FINNFOREST_USE_LOCAL_YOLO=1
source infra/env.sh
```

## 从空系统重建

```bash
cd /root/workSLAM
./infra/install_system.sh
./infra/fetch_sources.sh
./infra/install_python.sh
./infra/build_cpp.sh
./infra/verify.sh
```

构建脚本显式设置 `OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4`，防止 ORB-SLAM3 误链接 `/usr/local` 中的 OpenCV 4.7.0。不要删掉这一设置，也不要让 C++ 目标链接 Python wheel 的 OpenCV 5.0.0。

## 当前目录

```text
/root/workSLAM/
├── .venv-finnforest/
├── .local/
├── data/                 # 等待上传 FinnForest 数据
├── YOLO/                 # 等待上传用户工程和权重
├── configs/
├── downloads/            # 固定版本源码归档
├── infra/                # 安装、构建、环境和验收脚本
├── results/
├── scripts/
├── tests/
└── third_party/
    ├── Pangolin/
    └── ORB_SLAM3/
```

## 下一步需要人工上传

- FinnForest：至少 `W01_13Hz`，正式调参还需要 `S03/W03`；W01 是测试集，不能用于选阈值。
- `AllGroundTruths_and_Calibration.zip`、`toolkit.zip` 和需要使用的 ROS bag 数据。
- 用户 YOLO 工程或至少 `yolo_seg_fin3/weights/best.pt`。

推荐目标位置：

```text
/root/workSLAM/data/W01_13Hz
/root/workSLAM/YOLO
```

数据上传后先检查左右目帧数、零字节文件、时间戳和标定文件，再生成 FinnForest 的 ORB-SLAM3 双目配置。标定外参、左右目顺序、平移符号和毫米/米单位必须通过立体几何验证，不能直接照抄 YAML。
