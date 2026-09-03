#!/usr/bin/env bash
# Source this file from the repository root (or from any directory).
_ff_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FINNFOREST_ROOT="${_ff_root}"
export VIRTUAL_ENV="${_ff_root}/.venv-finnforest"
if [[ -f "${VIRTUAL_ENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VIRTUAL_ENV}/bin/activate"
fi

export CMAKE_PREFIX_PATH="${_ff_root}/.local${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${_ff_root}/.local/lib:${_ff_root}/third_party/ORB_SLAM3/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${_ff_root}/.local/bin:${PATH}"
if [[ "${FINNFOREST_USE_LOCAL_YOLO:-0}" == "1" ]]; then
  export PYTHONPATH="${_ff_root}/YOLO${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONNOUSERSITE=1

echo "FINNFOREST_ROOT=${FINNFOREST_ROOT}"
echo "python=$(command -v python || true)"
