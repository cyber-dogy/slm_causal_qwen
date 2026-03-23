#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/qwen_vis_fusion/configs/official_cv"
OUTPUT_ROOT="${PROJECT_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
REPORT_DIR="${OUTPUT_ROOT}/qwen_vis_fusion/reports/official_cv_last2"

if [[ -z "${SLM_DATA1_PROTOCOL_ROOT:-}" || -z "${QWEN2_VL_MODEL_PATH:-}" ]]; then
  echo "Please export SLM_DATA1_PROTOCOL_ROOT and QWEN2_VL_MODEL_PATH first."
  exit 1
fi

run_cfg() {
  local name="$1"
  python "${REPO_ROOT}/qwen_vis_fusion/train_cv.py" \
    --config "${CONFIG_DIR}/${name}.json"
}

run_cfg 3class_rgb_dual_last2blocks_baseline
run_cfg 3class_rgb_dual_last2blocks_causal

python "${REPO_ROOT}/qwen_vis_fusion/report.py" \
  --run_roots \
    "${OUTPUT_ROOT}/qwen_vis_fusion/official_cv_2class_rgb_dual_baseline" \
    "${OUTPUT_ROOT}/qwen_vis_fusion/official_cv_2class_rgb_dual_causal" \
    "${OUTPUT_ROOT}/qwen_vis_fusion/official_cv_3class_rgb_dual_baseline" \
    "${OUTPUT_ROOT}/qwen_vis_fusion/official_cv_3class_rgb_dual_causal" \
    "${OUTPUT_ROOT}/qwen_vis_fusion/official_cv_3class_rgb_dual_last2blocks_baseline" \
    "${OUTPUT_ROOT}/qwen_vis_fusion/official_cv_3class_rgb_dual_last2blocks_causal" \
  --output_dir "${REPORT_DIR}"
