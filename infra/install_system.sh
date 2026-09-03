#!/usr/bin/env bash
set -euo pipefail

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake git pkg-config \
  libopencv-dev libeigen3-dev libboost-all-dev libssl-dev \
  libglew-dev libgl1-mesa-dev libwayland-dev libxkbcommon-dev libegl1-mesa-dev \
  libpython3-dev python3-dev python3-venv python3-pip \
  unzip p7zip-full wget curl ffmpeg
