#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/qwen_vis_fusion/configs/official_cv"

if [[ -z "${SLM_DATA1_PROTOCOL_ROOT:-}" || -z "${QWEN2_VL_MODEL_PATH:-}" ]]; then
  echo "Please export SLM_DATA1_PROTOCOL_ROOT and QWEN2_VL_MODEL_PATH first."
  exit 1
fi

run_cfg() {
  local name="$1"
  python "${REPO_ROOT}/qwen_vis_fusion/train_cv.py" \
    --config "${CONFIG_DIR}/${name}.json"
}

run_cfg 3class_rgb_view1_baseline
run_cfg 3class_ir_only_baseline
run_cfg 3class_triple_view_baseline
