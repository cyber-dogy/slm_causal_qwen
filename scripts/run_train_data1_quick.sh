#!/bin/bash
set -euo pipefail

MODE=${1:-2class}
METHOD=${2:-baseline}
RUN_ROOT=${3:-./runs/data1_quick_dev_${MODE}_${METHOD}}
TOKEN_STYLE=${4:-letters}

EXTRA_ARGS=()
if [ "$#" -gt 4 ]; then
  EXTRA_ARGS=("${@:5}")
fi

./run_train_data1_protocol.sh quick_dev "$MODE" "$METHOD" "$RUN_ROOT" "$TOKEN_STYLE" all "${EXTRA_ARGS[@]}"
