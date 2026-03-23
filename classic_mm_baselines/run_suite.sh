#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${SLM_DATA1_PROTOCOL_ROOT:-}}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_OUTPUT_ROOT:-${REPO_ROOT}/outputs}/classic_mm_baselines}"
REPORT_DIR="${REPORT_DIR:-${PROJECT_OUTPUT_ROOT:-${REPO_ROOT}/outputs}/classic_mm_baselines/reports/official_cv}"

if [[ -z "${PROTOCOL_ROOT}" ]]; then
  echo "Please export PROTOCOL_ROOT or SLM_DATA1_PROTOCOL_ROOT first."
  exit 1
fi

mkdir -p "${RUNS_ROOT}"

run_exp() {
  local task_mode="$1"
  local fusion_type="$2"
  local run_root="${RUNS_ROOT}/official_cv_${task_mode}_${fusion_type}"

  python "${REPO_ROOT}/classic_mm_baselines/train_cv.py" \
    --protocol_root "${PROTOCOL_ROOT}" \
    --scope official_cv \
    --task_mode "${task_mode}" \
    --fusion_type "${fusion_type}" \
    --run_root "${run_root}" \
    --fold all
}

run_exp 2class single_view
run_exp 2class mlp_concat
run_exp 2class cross_attention
run_exp 3class single_view
run_exp 3class mlp_concat
run_exp 3class cross_attention

python "${REPO_ROOT}/classic_mm_baselines/report.py" \
  --run_roots \
    "${RUNS_ROOT}/official_cv_2class_single_view" \
    "${RUNS_ROOT}/official_cv_2class_mlp_concat" \
    "${RUNS_ROOT}/official_cv_2class_cross_attention" \
    "${RUNS_ROOT}/official_cv_3class_single_view" \
    "${RUNS_ROOT}/official_cv_3class_mlp_concat" \
    "${RUNS_ROOT}/official_cv_3class_cross_attention" \
  --output_dir "${REPORT_DIR}"
