#!/bin/bash
set -euo pipefail

MODE=${1:-2class}
METHOD=${2:-baseline}
RUN_ROOT=${3:-./runs/data1_official_cv_${MODE}_${METHOD}}
TOKEN_STYLE=${4:-letters}
FOLD=${5:-all}

EXTRA_ARGS=()
if [ "$#" -gt 5 ]; then
  EXTRA_ARGS=("${@:6}")
fi

./run_train_data1_protocol.sh official_cv "$MODE" "$METHOD" "$RUN_ROOT" "$TOKEN_STYLE" "$FOLD" "${EXTRA_ARGS[@]}"
