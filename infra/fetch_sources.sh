#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/downloads" "$root/third_party"

pangolin_archive="$root/downloads/Pangolin-v0.6.tar.gz"
orb_archive="$root/downloads/ORB_SLAM3-4452a3c.tar.gz"

if [[ ! -f "$pangolin_archive" ]]; then
  curl -fL --retry 10 --retry-delay 3 --retry-all-errors \
    -o "$pangolin_archive" \
    https://codeload.github.com/stevenlovegrove/Pangolin/tar.gz/refs/tags/v0.6
fi
if [[ ! -f "$orb_archive" ]]; then
  curl -fL --retry 10 --retry-delay 3 --retry-all-errors \
    -o "$orb_archive" \
    https://codeload.github.com/UZ-SLAMLab/ORB_SLAM3/tar.gz/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
fi

echo '2fe0f7242bee9b7241a73d17878fe25179f22b07a18d02dec069e4c90890e9d5  '"$pangolin_archive" | sha256sum -c -
echo 'b69fe1809588a8610580db1d5e6b77c63b5e4b5b47abc16ed839e93fbc7e7ddf  '"$orb_archive" | sha256sum -c -

if [[ ! -f "$root/third_party/Pangolin/CMakeLists.txt" ]]; then
  mkdir -p "$root/third_party/Pangolin"
  tar -xzf "$pangolin_archive" --strip-components=1 -C "$root/third_party/Pangolin"
fi
if [[ ! -f "$root/third_party/ORB_SLAM3/CMakeLists.txt" ]]; then
  mkdir -p "$root/third_party/ORB_SLAM3"
  tar -xzf "$orb_archive" --strip-components=1 -C "$root/third_party/ORB_SLAM3"
fi
