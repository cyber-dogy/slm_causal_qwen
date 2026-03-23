#!/usr/bin/env bash
set -euo pipefail

cd /home/gjw/code/Qwen-SLM

python data2_transfer_eval/build_transfer_protocol.py
python data2_transfer_eval/eval_prompt_qwen.py
python data2_transfer_eval/eval_classic.py
python data2_transfer_eval/eval_qwen_vis.py
python data2_transfer_eval/report.py
