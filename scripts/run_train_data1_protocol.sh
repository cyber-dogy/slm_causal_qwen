#!/bin/bash
set -euo pipefail

SCOPE=${1:-official_cv}
MODE=${2:-2class}
METHOD=${3:-baseline}
RUN_ROOT=${4:-./runs/data1_${SCOPE}_${MODE}_${METHOD}}
TOKEN_STYLE=${5:-letters}
FOLD=${6:-all}
PROTOCOL_ROOT=${PROTOCOL_ROOT:-/home/gjw/code/SLM_data/processed_qwen_data1_protocol}
MODEL_PATH=${MODEL_PATH:-/home/gjw/code/model_cache/Qwen/Qwen2-VL-2B-Instruct}
WANDB_PROJECT=${WANDB_PROJECT:-qwen2vl-data1}

EXTRA_ARGS=()
if [ "$#" -gt 6 ]; then
  EXTRA_ARGS=("${@:7}")
fi

if [ "$SCOPE" != "quick_dev" ] && [ "$SCOPE" != "official_cv" ]; then
  echo "Unsupported scope: $SCOPE (expected quick_dev or official_cv)"
  exit 1
fi

if [ "$MODE" != "2class" ] && [ "$MODE" != "3class" ]; then
  echo "Unsupported mode: $MODE (expected 2class or 3class)"
  exit 1
fi

if [ "$METHOD" = "baseline" ]; then
  TASK_MODE=naive_sft
  CAUSAL_LOSS_WEIGHT=0.0
elif [ "$METHOD" = "causal" ]; then
  TASK_MODE=causal_train
  CAUSAL_LOSS_WEIGHT=${CAUSAL_LOSS_WEIGHT:-0.1}
else
  echo "Unsupported method: $METHOD (expected baseline or causal)"
  exit 1
fi

LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.10}
LEARNING_RATE=${LEARNING_RATE:-0.00002}
NUM_EPOCHS=${NUM_EPOCHS:-15}
EARLY_STOPPING=${EARLY_STOPPING:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}

run_single_fold () {
  local fold_name="$1"
  local train_manifest="$2"
  local val_manifest="$3"
  local output_dir="$4"

  local wandb_run_name
  wandb_run_name="$(basename "$RUN_ROOT")_${fold_name}_${MODE}_${METHOD}"

  echo "训练: scope=$SCOPE fold=$fold_name mode=$MODE method=$METHOD output=$output_dir"

  python src/train_qwen2vl_slm_lora.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_manifest "$train_manifest" \
    --val_manifest "$val_manifest" \
    --output_dir "$output_dir" \
    --overwrite_output_dir true \
    --target_mode "$MODE" \
    --token_style "$TOKEN_STYLE" \
    --task_mode "$TASK_MODE" \
    --prediction_mode auto \
    --compute_candidate_scores true \
    --selection_metric auto \
    --causal_loss_weight "$CAUSAL_LOSS_WEIGHT" \
    --finetune_mode qlora \
    --load_in_4bit true \
    --bf16 true \
    --num_train_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0.02 \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --early_stopping_patience "$EARLY_STOPPING" \
    --max_grad_norm 0.5 \
    --max_new_tokens 4 \
    --seed 42 \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$wandb_run_name" \
    "${EXTRA_ARGS[@]}"
}

if [ "$SCOPE" = "quick_dev" ]; then
  run_single_fold \
    quick_dev \
    "$PROTOCOL_ROOT/quick_dev/train_manifest.jsonl" \
    "$PROTOCOL_ROOT/quick_dev/val_manifest.jsonl" \
    "$RUN_ROOT"
  exit 0
fi

if [ "$FOLD" = "all" ]; then
  FOLD_DIRS=($(find "$PROTOCOL_ROOT/official_cv" -maxdepth 1 -mindepth 1 -type d -name 'fold_*' | sort))
else
  FOLD_DIRS=("$PROTOCOL_ROOT/official_cv/$FOLD")
fi

for fold_dir in "${FOLD_DIRS[@]}"; do
  if [ ! -d "$fold_dir" ]; then
    echo "Missing fold directory: $fold_dir"
    exit 1
  fi

  fold_name="$(basename "$fold_dir")"
  run_single_fold \
    "$fold_name" \
    "$fold_dir/train_manifest.jsonl" \
    "$fold_dir/val_manifest.jsonl" \
    "$RUN_ROOT/$fold_name"
done
