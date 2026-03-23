#!/bin/bash
set -euo pipefail

OUTPUT_ROOT=${1:-/home/gjw/code/SLM_data/processed_qwen_data1_protocol}
EXTRA_ARGS=()
if [ "$#" -gt 1 ]; then
  EXTRA_ARGS=("${@:2}")
fi

echo "构建 data1 grouped protocol: output_root=$OUTPUT_ROOT"

python src/dataset_qwen2vl_slm.py \
  --build_data1_protocol \
  --data1_dir /home/gjw/code/SLM_data/Causal_Image_Data \
  --protocol_output_root "$OUTPUT_ROOT" \
  --official_num_folds 3 \
  --official_val_ratio 0.25 \
  --quick_val_ratio 0.20 \
  --quick_test_ratio 0.20 \
  --protocol_balance_train true \
  --protocol_balance_mode upsample_to_max \
  --seed 42 \
  "${EXTRA_ARGS[@]}"
