#!/usr/bin/env bash
set -euo pipefail

# Download and extract Objects365-2020 image patches without using proxy traffic.
#
# Defaults download the full patch set expected by CLOC annotations:
#   train/patch0 ... patch50
#   val/images/v1/patch0 ... patch15
#   val/images/v2/patch16 ... patch43
#
# For a small smoke test:
#   TRAIN_PATCHES="0" VAL_V1_PATCHES="0" VAL_V2_PATCHES="16" \
#     bash tools/downloaders/download_objects365_patches_no_proxy.sh

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ROOT="${OBJECTS365_ROOT:-${WORKSPACE}/images/Objects365-2020}"
BASE="${OBJECTS365_BASE:-https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

TRAIN_PATCHES="${TRAIN_PATCHES:-$(seq 0 50 | tr '\n' ' ')}"
VAL_V1_PATCHES="${VAL_V1_PATCHES:-$(seq 0 15 | tr '\n' ' ')}"
VAL_V2_PATCHES="${VAL_V2_PATCHES:-$(seq 16 43 | tr '\n' ' ')}"

normalize_patch_list() {
  echo "$1" | tr ',' ' '
}

download_extract_patch() {
  local kind="$1"
  local patch="$2"
  local url="$3"
  local extract_dir="$4"
  local out="$extract_dir/patch${patch}.tar.gz"
  local expected_dir="$extract_dir/patch${patch}"

  mkdir -p "$extract_dir"

  if [[ -d "$expected_dir" ]] && find "$expected_dir" -type f -name '*.jpg' -print -quit | grep -q .; then
    echo "[skip] $kind patch${patch}: already extracted at $expected_dir"
    return
  fi

  echo "[download] $kind patch${patch}"
  curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 -o "$out" "$url"

  echo "[extract] $kind patch${patch}"
  tar -xzf "$out" -C "$extract_dir"

  if [[ "$KEEP_ARCHIVES" != "1" ]]; then
    rm -f "$out"
  fi

  if [[ ! -d "$expected_dir" ]]; then
    echo "[error] expected extracted directory not found: $expected_dir" >&2
    return 1
  fi
}

echo "Objects365 root: $ROOT"
echo "Base URL: $BASE"

for i in $(normalize_patch_list "$TRAIN_PATCHES"); do
  [[ -z "$i" ]] && continue
  download_extract_patch "train" "$i" "$BASE/train/patch${i}.tar.gz" "$ROOT/train"
done

for i in $(normalize_patch_list "$VAL_V1_PATCHES"); do
  [[ -z "$i" ]] && continue
  download_extract_patch "val-v1" "$i" "$BASE/val/images/v1/patch${i}.tar.gz" "$ROOT/val/images/v1"
done

for i in $(normalize_patch_list "$VAL_V2_PATCHES"); do
  [[ -z "$i" ]] && continue
  download_extract_patch "val-v2" "$i" "$BASE/val/images/v2/patch${i}.tar.gz" "$ROOT/val/images/v2"
done

echo "[done] Objects365 patch download/extract completed."
