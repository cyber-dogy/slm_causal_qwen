# Data2 Transfer Eval

这个目录与现有 `data1 official_cv`、`classic_mm_baselines`、`qwen_vis_fusion` 主线解耦，专门用于：

- 读取并分析 `data2`
- 构建独立的 `transfer_test` manifest
- 将此前训练好的所有模型迁移到 `data2`
- 生成统一的 transfer 报告

默认不会改动原有训练脚本或旧报告，只在新的输出目录写文件。

## 输出位置

- transfer manifest:
  `/home/gjw/code/SLM_data/processed_qwen_data2_transfer`
- 评估输出:
  `/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval`

## 一键运行

```bash
python data2_transfer_eval/build_transfer_protocol.py
python data2_transfer_eval/eval_prompt_qwen.py
python data2_transfer_eval/eval_classic.py
python data2_transfer_eval/eval_qwen_vis.py
python data2_transfer_eval/report.py
```

或直接：

```bash
./data2_transfer_eval/run_all.sh
```

## 主要产物

- `transfer_test_manifest.jsonl`
- `data2_analysis.md`
- `prompt_qwen_summary.json`
- `classic_summary.json`
- `qwen_vis_summary.json`
- `comparison_report.md`
- `comparison_report.zh_en.md`

