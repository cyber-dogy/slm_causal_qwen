#!/bin/bash
set -euo pipefail

SCOPE=${1:-official_cv}
STAGE=${2:-both}
MODE=${3:-all}
SPLIT=${4:-all}
RUN_ROOT_2CLASS=${5:-./runs/data1_${SCOPE}_2class_baseline}
RUN_ROOT_3CLASS_BASELINE=${6:-./runs/data1_${SCOPE}_3class_baseline}
RUN_ROOT_3CLASS_CAUSAL=""
FOLD="all"

if [ "$#" -ge 7 ]; then
  if [ "${7}" = "all" ] || [[ "${7}" == fold_* ]]; then
    FOLD="${7}"
    EXTRA_START=8
  else
    RUN_ROOT_3CLASS_CAUSAL="${7}"
    FOLD=${8:-all}
    EXTRA_START=9
  fi
else
  EXTRA_START=8
fi

EXTRA_ARGS=()
if [ "$#" -ge "$EXTRA_START" ]; then
  EXTRA_ARGS=("${@:${EXTRA_START}}")
fi

echo "评估 data1 protocol: scope=$SCOPE stage=$STAGE mode=$MODE split=$SPLIT fold=$FOLD"
echo "  run_root_2class=$RUN_ROOT_2CLASS"
echo "  run_root_3class_baseline=$RUN_ROOT_3CLASS_BASELINE"
if [ -n "$RUN_ROOT_3CLASS_CAUSAL" ]; then
  echo "  run_root_3class_causal=$RUN_ROOT_3CLASS_CAUSAL"
fi

CMD=(
  python evaluate_data1_protocol.py
  --scope "$SCOPE"
  --stage "$STAGE"
  --mode "$MODE"
  --split "$SPLIT"
  --fold "$FOLD"
  --protocol_root /home/gjw/code/SLM_data/processed_qwen_data1_protocol
  --run_root_2class "$RUN_ROOT_2CLASS"
  --run_root_3class_baseline "$RUN_ROOT_3CLASS_BASELINE"
  --model /home/gjw/code/model_cache/Qwen/Qwen2-VL-2B-Instruct
  --token_style letters
  --prediction_mode candidate_score
  --compute_candidate_scores true
)

if [ -n "$RUN_ROOT_3CLASS_CAUSAL" ]; then
  CMD+=(--run_root_3class_causal "$RUN_ROOT_3CLASS_CAUSAL")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
