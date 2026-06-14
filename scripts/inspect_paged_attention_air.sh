#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT}/build/air}"
METAL_SRC_DIR="${ROOT}/paged_attention/metal"
MODULE_CACHE="${OUT_DIR}/module-cache"

mkdir -p "${OUT_DIR}"
mkdir -p "${MODULE_CACHE}"

xcrun -sdk macosx metal \
  -fmodules-cache-path="${MODULE_CACHE}" \
  -I "${METAL_SRC_DIR}" \
  -Wall -Wextra -fno-fast-math -Wno-c++17-extensions \
  -gline-tables-only -frecord-sources=flat \
  -c "${METAL_SRC_DIR}/attention/paged_attention_inst.metal" \
  -o "${OUT_DIR}/paged_attention.air"

if command -v metal-dis >/dev/null 2>&1; then
  metal-dis "${OUT_DIR}/paged_attention.air" > "${OUT_DIR}/paged_attention.air.txt"
  echo "Wrote ${OUT_DIR}/paged_attention.air.txt"
else
  echo "Wrote ${OUT_DIR}/paged_attention.air"
  echo "metal-dis was not found; install Xcode command line tools that include it to dump text AIR."
fi
