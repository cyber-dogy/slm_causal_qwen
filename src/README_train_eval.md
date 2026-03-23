# Qwen2-VL SLM Defect Detection - Training & Evaluation

A complete, production-ready implementation for fine-tuning and evaluating Qwen2-VL-2B-Instruct on SLM (Selective Laser Melting) industrial defect detection tasks.

## Features

- **Parameter-Efficient Fine-Tuning**: LoRA and QLoRA support for training on consumer GPUs
- **Comprehensive Evaluation**: Before/after comparison, subset evaluation, transfer gap analysis
- **Safety First**: Guaranteed protection against overwriting original model weights
- **Candidate Scoring**: Probability-based evaluation using teacher-forced likelihood
- **Extensible Design**: Reserved interfaces for before/after consistency loss

## Project Structure

```
Qwen-SLM/
├── src/
│   ├── train_qwen2vl_slm_lora.py    # Training script
│   ├── evaluate_qwen2vl_slm.py      # Evaluation script
│   ├── dataset_qwen2vl_slm.py       # Dataset implementation
│   ├── utils_metrics.py             # Metrics computation
│   ├── run_train_eval.ps1           # PowerShell pipeline
│   ├── requirements.txt             # Python dependencies
│   └── README_train_eval.md         # This file
└── runs/
    └── run_YYYYMMDD_HHMMSS/         # Individual run outputs
        ├── config/                  # Training configuration
        ├── checkpoints/             # Periodic checkpoints
        ├── adapter_best/            # Best LoRA adapter
        ├── eval_before/             # Pre-training evaluation
        │   ├── val/
        │   └── test/
        ├── eval_after/              # Post-training evaluation
        │   ├── val/
        │   └── test/
        ├── predictions/             # Prediction outputs
        ├── reports/                 # Comparison reports
        └── logs/                    # Training logs
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```powershell
# PowerShell
.\run_train_eval.ps1 `
    -RunName "my_experiment" `
    -BaseModel "Qwen/Qwen2-VL-2B-Instruct" `
    -DataDir "/home/gjw/code/SLM_data/processed" `
    -FinetuneMode "qlora" `
    -NumEpochs 10
```

### 3. View Results

Check the output directory for:
- `reports/before_after_comparison.md` - Comprehensive comparison
- `adapter_best/` - Best LoRA adapter for inference
- `eval_after/test/metrics_summary.json` - Final test metrics

## Detailed Usage

### Training Only

```bash
python train_qwen2vl_slm_lora.py \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --train_manifest "/home/gjw/code/SLM_data/processed/train_manifest.jsonl" \
    --val_manifest "/home/gjw/code/SLM_data/processed/val_manifest.jsonl" \
    --test_manifest "/home/gjw/code/SLM_data/processed/test_manifest.jsonl" \
    --output_dir "/home/gjw/code/Qwen-SLM/runs/experiment_001" \
    --finetune_mode qlora \
    --load_in_4bit true \
    --bf16 true \
    --num_train_epochs 10 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --early_stopping_patience 3
```

### Evaluation Only (Base Model)

```bash
python evaluate_qwen2vl_slm.py \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --manifest_path "/home/gjw/code/SLM_data/processed/test_manifest.jsonl" \
    --output_dir "/home/gjw/code/Qwen-SLM/runs/experiment_001/eval_before/test" \
    --eval_name "test_before" \
    --compute_candidate_scores true \
    --evaluate_subsets true
```

### Evaluation with Adapter

```bash
python evaluate_qwen2vl_slm.py \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --adapter_path "/home/gjw/code/Qwen-SLM/runs/experiment_001/adapter_best" \
    --manifest_path "/home/gjw/code/SLM_data/processed/test_manifest.jsonl" \
    --output_dir "/home/gjw/code/Qwen-SLM/runs/experiment_001/eval_after/test" \
    --eval_name "test_after" \
    --compute_candidate_scores true
```

## Hardware Configuration Guide

### 24GB VRAM (e.g., RTX 3090, 4090)

```bash
--finetune_mode qlora \
--load_in_4bit true \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--bf16 true
```

### 48GB+ VRAM (e.g., A6000, A100)

```bash
--finetune_mode lora \
--load_in_4bit false \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--bf16 true
```

### CPU Only (Not Recommended)

Training on CPU is extremely slow. Use evaluation mode only:

```bash
--load_in_4bit false \
--bf16 false
```

## Key Parameters

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--finetune_mode` | `qlora` | `lora` or `qlora` |
| `--load_in_4bit` | `true` | Use 4-bit quantization |
| `--num_train_epochs` | `10` | Maximum training epochs |
| `--learning_rate` | `2e-4` | Learning rate |
| `--gradient_accumulation_steps` | `8` | Gradient accumulation |
| `--early_stopping_patience` | `3` | Early stopping patience |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--compute_candidate_scores` | `true` | Compute probability scores |
| `--evaluate_subsets` | `true` | Evaluate domain subsets |
| `--max_new_tokens` | `4` | Max tokens to generate |
| `--batch_size` | `1` | Evaluation batch size |

## Output Files

### Training Outputs

- `adapter_best/` - Best LoRA adapter (based on validation macro F1)
- `checkpoints/checkpoint-epoch-{N}/` - Periodic checkpoints
- `logs/train.log` - Training log
- `logs/train_log.csv` - Metrics per epoch
- `config/train_config.json` - Training configuration

### Evaluation Outputs

- `metrics_summary.json` - All computed metrics
- `metrics_summary.csv` - Metrics in CSV format
- `metrics_per_class.csv` - Per-class precision/recall/F1
- `confusion_matrix.csv` - Confusion matrix
- `confusion_matrix.png` - Confusion matrix visualization
- `predictions.jsonl` - Detailed predictions per sample
- `prediction_errors.csv` - Incorrect predictions only

## Safety Guarantees

This implementation has multiple safeguards to prevent overwriting the original model:

1. **Path Validation**: Output directory cannot be the same as or inside the base model path
2. **Read-Only Base**: Base model is always loaded in read-only mode
3. **Adapter-Only Saving**: Only LoRA adapters are saved, never full model weights
4. **Explicit Overwrite**: Existing output directories are protected unless `--overwrite_output_dir true` is set

Example safety check:
```python
# This will raise an error
python train_qwen2vl_slm_lora.py \
    --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
    --output_dir "Qwen/Qwen2-VL-2B-Instruct"  # DANGER! Same as base model
```

## Metrics Explanation

### Standard Metrics

- **Accuracy**: Overall correct predictions / total
- **Balanced Accuracy**: Average of per-class recall (handles imbalance)
- **Macro F1**: Unweighted mean of per-class F1 scores
- **Weighted F1**: F1 weighted by class support

### Transfer Gap Metrics

- **Transfer Gap Macro F1**: `val_macro_f1 - test_macro_f1`
  - Measures overfitting to source domain
  - Smaller is better
- **Transfer Gap Balanced Accuracy**: Similar for balanced accuracy

### Candidate Probability Metrics

When `--compute_candidate_scores true`:

- **ROC-AUC**: Area under ROC curve (one-vs-rest)
- **PR-AUC**: Area under precision-recall curve
- Helps identify model confidence vs. calibration

## Troubleshooting

### CUDA Out of Memory

1. Increase gradient accumulation: `--gradient_accumulation_steps 16`
2. Use QLoRA: `--finetune_mode qlora --load_in_4bit true`
3. Reduce batch size: `--per_device_train_batch_size 1`

### Slow Training

1. Check GPU utilization: `nvidia-smi`
2. Enable gradient checkpointing (automatic)
3. Use flash attention if available (install `flash-attn`)

### Poor Performance

1. Check learning rate: Try 1e-4 to 5e-4
2. Increase LoRA rank: `--lora_r 32`
3. Add more target modules (automatically detected)
4. Check class distribution in training data

## Extending for Causal Consistency Loss

The code is designed to support future extensions for before/after consistency:

### Current State

- `images_before` is loaded in Dataset but not used
- `--use_before_aux` flag exists (default: false)

### Future Extension Points

1. **Dataset**: Load before images when `use_before_aux=true`
2. **Model**: Add auxiliary branch for before image encoding
3. **Loss**: Implement `compute_aux_consistency_loss()` function
4. **Training**: Add consistency loss to total loss

Example future usage:
```bash
python train_qwen2vl_slm_lora.py \
    --use_before_aux true \
    --consistency_loss_weight 0.5
```

## Data Format

### Manifest Format (train_manifest.jsonl)

```json
{
  "id": "data1__cond_001__L0001",
  "split": "train",
  "images_after": [
    "path/to/rgb_view1_after.png",
    "path/to/rgb_view2_after.png",
    "path/to/ir_view_after.png"
  ],
  "images_before": [
    "path/to/rgb_view1_before.png",
    "path/to/rgb_view2_before.png",
    "path/to/ir_view_before.png"
  ],
  "label": "HEW",
  "label_id": 1,
  "prompt": "Given these three monitoring images...",
  "metadata": {
    "dataset_domain": "data1",
    "shape_domain": "cube",
    "condition_id": "cond_001",
    "role": "seen"
  }
}
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qwen2vl_slm_detection,
  title={Qwen2-VL SLM Defect Detection},
  author={Your Name},
  year={2024},
  note={Fine-tuning and evaluation framework for industrial defect detection}
}
```

## License

This project is provided as-is for research and educational purposes.

## Support

For issues and questions:
1. Check this README
2. Review training logs in `logs/train.log`
3. Check evaluation outputs in `eval_*/**/metrics_summary.json`

## Acknowledgments

- Qwen2-VL: https://github.com/QwenLM/Qwen2-VL
- PEFT: https://github.com/huggingface/peft
- Transformers: https://github.com/huggingface/transformers
