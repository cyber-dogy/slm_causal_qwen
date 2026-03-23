#!/bin/bash
set -euo pipefail

STAGE=${1:-both}
MODE=${2:-all}
SUBSET=${3:-all}
RUN_DIR=${4:-./runs/exp_balanced_3class}
TOKEN_STYLE=${5:-letters}
EXTRA_ARGS=()
if [ "$#" -gt 5 ]; then
  EXTRA_ARGS=("${@:6}")
fi

echo "评估: stage=$STAGE, mode=$MODE, subset=$SUBSET, run_dir=$RUN_DIR, token_style=$TOKEN_STYLE"

python evaluate_balanced.py \
  --stage "$STAGE" \
  --mode "$MODE" \
  --subset "$SUBSET" \
  --run_dir "$RUN_DIR" \
  --data_dir /home/gjw/code/SLM_data/processed_qwen_balanced \
  --model /home/gjw/code/model_cache/Qwen/Qwen2-VL-2B-Instruct \
  --token_style "$TOKEN_STYLE" \
  --prediction_mode candidate_score \
  --compute_candidate_scores true \
  "${EXTRA_ARGS[@]}"
