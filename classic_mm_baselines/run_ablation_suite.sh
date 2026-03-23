#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/classic_mm_baselines/configs/official_cv"
OUTPUT_ROOT="${PROJECT_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
REPORT_DIR="${OUTPUT_ROOT}/classic_mm_baselines/reports/official_cv_extended"

if [[ -z "${SLM_DATA1_PROTOCOL_ROOT:-}" ]]; then
  echo "Please export SLM_DATA1_PROTOCOL_ROOT first."
  exit 1
fi

run_cfg() {
  local name="$1"
  python "${REPO_ROOT}/classic_mm_baselines/train_cv.py" \
    --config "${CONFIG_DIR}/${name}.json"
}

run_cfg 2class_ir_only_single_view
run_cfg 2class_rgb_dual_mlp_concat
run_cfg 2class_triple_view_late_fusion
run_cfg 3class_ir_only_single_view
run_cfg 3class_rgb_dual_mlp_concat
run_cfg 3class_triple_view_late_fusion

python "${REPO_ROOT}/classic_mm_baselines/report.py" \
  --run_roots \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_single_view" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_mlp_concat" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_cross_attention" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_ir_only_single_view" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_rgb_dual_mlp_concat" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_triple_view_late_fusion" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_single_view" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_mlp_concat" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_cross_attention" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_ir_only_single_view" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_rgb_dual_mlp_concat" \
    "${OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_triple_view_late_fusion" \
  --output_dir "${REPORT_DIR}"
