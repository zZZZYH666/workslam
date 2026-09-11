#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
source infra/env.sh >/dev/null
python -m pip check
python - <<'PY'
import cv2, torch, ultralytics, evo
from rosbags.highlevel import AnyReader
print('torch:', torch.__version__)
print('torch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
print('python OpenCV:', cv2.__version__)
print('ultralytics:', ultralytics.__version__)
print('evo:', evo.__version__)
print('rosbags import: OK')
PY
test "$(pkg-config --modversion opencv4)" = 4.5.4
echo "C++ OpenCV: $(pkg-config --modversion opencv4)"
if [[ -x third_party/ORB_SLAM3/Examples/Stereo/stereo_tum_vi ]]; then
  if ldd third_party/ORB_SLAM3/Examples/Stereo/stereo_tum_vi | grep -q 'not found'; then
    echo 'ORB-SLAM3 dynamic library check failed' >&2; exit 1
  fi
  echo 'ORB-SLAM3 dynamic library check: OK'
else
  echo 'ORB-SLAM3 executable not built yet' >&2; exit 2
fi
