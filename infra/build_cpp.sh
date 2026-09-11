#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
jobs="${BUILD_JOBS:-2}"
opencv_dir="/usr/lib/x86_64-linux-gnu/cmake/opencv4"
cd "$root"
source infra/env.sh >/dev/null

cmake -S third_party/Pangolin -B third_party/Pangolin/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$root/.local" \
  -DBUILD_PANGOLIN_PYTHON=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TOOLS=OFF
cmake --build third_party/Pangolin/build -j"$jobs"
cmake --install third_party/Pangolin/build

orb="$root/third_party/ORB_SLAM3"
cmake -S "$orb/Thirdparty/DBoW2" -B "$orb/Thirdparty/DBoW2/build" \
  -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$opencv_dir"
cmake --build "$orb/Thirdparty/DBoW2/build" -j"$jobs"
cmake -S "$orb/Thirdparty/g2o" -B "$orb/Thirdparty/g2o/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$orb/Thirdparty/g2o/build" -j"$jobs"
cmake -S "$orb/Thirdparty/Sophus" -B "$orb/Thirdparty/Sophus/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$orb/Thirdparty/Sophus/build" -j"$jobs"

if [[ ! -f "$orb/Vocabulary/ORBvoc.txt" ]]; then
  tar -xf "$orb/Vocabulary/ORBvoc.txt.tar.gz" -C "$orb/Vocabulary"
fi
cmake -S "$orb" -B "$orb/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$root/.local" \
  -DOpenCV_DIR="$opencv_dir"
cmake --build "$orb/build" -j"$jobs"
