#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
if [[ ! -x .venv-finnforest/bin/python ]]; then
  python3 -m venv .venv-finnforest
fi
PIP_CONFIG_FILE=/dev/null .venv-finnforest/bin/python -m pip install --upgrade pip setuptools wheel
PIP_CONFIG_FILE=/dev/null .venv-finnforest/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128
PIP_CONFIG_FILE=/dev/null .venv-finnforest/bin/python -m pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  -r requirements.txt
