#!/bin/bash
set -euo pipefail

MODE=${1:-3class}
RUN_DIR=${2:-./runs/exp_balanced_${MODE}}
TOKEN_STYLE=${3:-letters}
WANDB_PROJECT=${WANDB_PROJECT:-qwen2vl-balanced}
EXTRA_ARGS=()
if [ "$#" -gt 3 ]; then
  EXTRA_ARGS=("${@:4}")
fi

if [ "$MODE" != "2class" ] && [ "$MODE" != "3class" ]; then
  echo "Unsupported mode: $MODE (expected 2class or 3class)"
  exit 1
fi

echo "训练: mode=$MODE, run_dir=$RUN_DIR, token_style=$TOKEN_STYLE"

python src/train_qwen2vl_slm_lora.py \
  --model_name_or_path /home/gjw/code/model_cache/Qwen/Qwen2-VL-2B-Instruct \
  --train_manifest /home/gjw/code/SLM_data/processed_qwen_balanced/train_manifest.jsonl \
  --val_manifest /home/gjw/code/SLM_data/processed_qwen_balanced/val_manifest.jsonl \
  --output_dir "$RUN_DIR" \
  --overwrite_output_dir true \
  --target_mode "$MODE" \
  --token_style "$TOKEN_STYLE" \
  --task_mode naive_sft \
  --prediction_mode auto \
  --compute_candidate_scores true \
  --selection_metric auto \
  --finetune_mode qlora \
  --load_in_4bit true \
  --bf16 true \
  --num_train_epochs 15 \
  --learning_rate 0.00002 \
  --weight_decay 0.02 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --early_stopping_patience 4 \
  --max_grad_norm 0.5 \
  --seed 42 \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$(basename "$RUN_DIR")_${MODE}" \
  "${EXTRA_ARGS[@]}"
