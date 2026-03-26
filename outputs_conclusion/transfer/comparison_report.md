# Data2 Transfer Comparison Report

Generated: 2026-03-23 16:30:23

## Data2 Overview

- `data2` is treated as a harder external transfer setting with unseen model/process shifts.
- `data1` remains the condition-heldout closed-set source protocol.

| Domain | Samples | Conditions | normal | HEW | LEL |
|--------|---------|------------|--------|-----|-----|
| data1 | 289 | 22 | 104 | 86 | 99 |
| data2 | 404 | 22 | 167 | 128 | 109 |

## Coverage

| Family | Experiment | Task | Method | Input | Stage | Folds | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Folded 2Class Macro F1 |
|--------|------------|------|--------|-------|-------|-------|----------|--------------|----------|-------------|------------------------|
| Classic | official_cv_2class_cross_attention | 2class | baseline | triple_view | after | 3 | 0.5916 ± 0.0592 | 0.5865 ± 0.0106 | 0.5435 ± 0.0564 | 0.5509 ± 0.0809 | - |
| Classic | official_cv_2class_mlp_concat | 2class | baseline | triple_view | after | 3 | 0.5924 ± 0.0459 | 0.5373 ± 0.0225 | 0.4788 ± 0.0508 | 0.5171 ± 0.0363 | - |
| Classic | official_cv_2class_single_view | 2class | baseline | triple_view | after | 3 | 0.4860 ± 0.0670 | 0.4646 ± 0.0189 | 0.4161 ± 0.0330 | 0.4356 ± 0.0562 | - |
| Classic | official_cv_3class_cross_attention | 3class | baseline | triple_view | after | 3 | 0.4241 ± 0.0117 | 0.3873 ± 0.0726 | 0.2721 ± 0.1013 | 0.2999 ± 0.0747 | 0.3853 ± 0.1246 |
| Classic | official_cv_3class_mlp_concat | 3class | baseline | triple_view | after | 3 | 0.4447 ± 0.0593 | 0.4036 ± 0.0264 | 0.3475 ± 0.0261 | 0.3709 ± 0.0395 | 0.4165 ± 0.0448 |
| Classic | official_cv_3class_single_view | 3class | baseline | triple_view | after | 3 | 0.4422 ± 0.0709 | 0.4291 ± 0.0948 | 0.3939 ± 0.0919 | 0.4031 ± 0.0767 | 0.4517 ± 0.0562 |
| Classic | resnet18_ir_only_single_view_2class | 2class | baseline | ir_only | after | 3 | 0.6229 ± 0.0303 | 0.5457 ± 0.0381 | 0.4619 ± 0.0745 | 0.5128 ± 0.0640 | - |
| Classic | resnet18_ir_only_single_view_3class | 3class | baseline | ir_only | after | 3 | 0.4431 ± 0.0073 | 0.4300 ± 0.0103 | 0.3522 ± 0.0234 | 0.3788 ± 0.0176 | 0.6180 ± 0.0213 |
| Classic | resnet18_rgb_dual_mlp_concat_2class | 2class | baseline | rgb_dual | after | 3 | 0.6155 ± 0.0304 | 0.5585 ± 0.0163 | 0.5104 ± 0.0452 | 0.5475 ± 0.0320 | - |
| Classic | resnet18_rgb_dual_mlp_concat_3class | 3class | baseline | rgb_dual | after | 3 | 0.4175 ± 0.0031 | 0.3455 ± 0.0125 | 0.2306 ± 0.0384 | 0.2694 ± 0.0293 | 0.3187 ± 0.0285 |
| Classic | resnet18_triple_view_late_fusion_2class | 2class | baseline | triple_view | after | 3 | 0.6452 ± 0.0294 | 0.5909 ± 0.0503 | 0.5523 ± 0.0986 | 0.5865 ± 0.0817 | - |
| Classic | resnet18_triple_view_late_fusion_3class | 3class | baseline | triple_view | after | 3 | 0.5231 ± 0.0678 | 0.4973 ± 0.0954 | 0.4236 ± 0.1110 | 0.4445 ± 0.0932 | 0.5136 ± 0.1331 |
| Prompt-Qwen | data1_official_cv_2class_baseline | 2class | baseline | triple_view | after | 3 | 0.5264 ± 0.0852 | 0.4958 ± 0.0059 | 0.3471 ± 0.0320 | 0.3742 ± 0.0842 | - |
| Prompt-Qwen | data1_official_cv_2class_baseline | 2class | baseline | triple_view | before | 3 | 0.4134 ± 0.0000 | 0.5000 ± 0.0000 | 0.2925 ± 0.0000 | 0.2418 ± 0.0000 | - |
| Prompt-Qwen | data1_official_cv_3class_baseline | 3class | baseline | triple_view | after | 3 | 0.4752 ± 0.0105 | 0.4586 ± 0.0286 | 0.4073 ± 0.0511 | 0.4201 ± 0.0358 | 0.5031 ± 0.0132 |
| Prompt-Qwen | data1_official_cv_3class_baseline | 3class | baseline | triple_view | before | 3 | 0.4134 ± 0.0000 | 0.3333 ± 0.0000 | 0.1950 ± 0.0000 | 0.2418 ± 0.0000 | 0.2925 ± 0.0000 |
| Prompt-Qwen | data1_official_cv_3class_causal | 3class | causal | triple_view | after | 3 | 0.4530 ± 0.0265 | 0.4116 ± 0.0274 | 0.3476 ± 0.0562 | 0.3748 ± 0.0522 | 0.4556 ± 0.0434 |
| Prompt-Qwen | data1_official_cv_3class_causal | 3class | causal | triple_view | before | 3 | 0.4134 ± 0.0000 | 0.3333 ± 0.0000 | 0.1950 ± 0.0000 | 0.2418 ± 0.0000 | 0.2925 ± 0.0000 |
| QwenVisFusion | official_cv_2class_rgb_dual_baseline | 2class | baseline | rgb_dual | after | 3 | 0.5850 ± 0.0023 | 0.4992 ± 0.0012 | 0.3727 ± 0.0041 | 0.4359 ± 0.0030 | - |
| QwenVisFusion | official_cv_2class_rgb_dual_causal | 2class | causal | rgb_dual | after | 3 | 0.5998 ± 0.0187 | 0.5289 ± 0.0409 | 0.4401 ± 0.0995 | 0.4908 ± 0.0806 | - |
| QwenVisFusion | official_cv_3class_rgb_dual_baseline | 3class | baseline | rgb_dual | after | 3 | 0.6304 ± 0.0320 | 0.6303 ± 0.0235 | 0.5964 ± 0.0494 | 0.5901 ± 0.0511 | 0.6302 ± 0.0321 |
| QwenVisFusion | official_cv_3class_rgb_dual_causal | 3class | causal | rgb_dual | after | 3 | 0.6386 ± 0.0210 | 0.6359 ± 0.0158 | 0.6067 ± 0.0351 | 0.6009 ± 0.0361 | 0.6384 ± 0.0211 |
| QwenVisFusion | official_cv_3class_rgb_dual_last2blocks_baseline | 3class | baseline | rgb_dual | after | 3 | 0.5899 ± 0.0970 | 0.6188 ± 0.0928 | 0.5684 ± 0.1272 | 0.5546 ± 0.1275 | 0.5811 ± 0.0942 |
| QwenVisFusion | official_cv_3class_rgb_dual_last2blocks_baseline | 3class | baseline | rgb_dual | before | 3 | 0.6304 ± 0.0320 | 0.6303 ± 0.0235 | 0.5964 ± 0.0494 | 0.5901 ± 0.0511 | 0.6302 ± 0.0321 |
| QwenVisFusion | official_cv_3class_rgb_dual_last2blocks_causal | 3class | causal | rgb_dual | after | 3 | 0.6106 ± 0.0296 | 0.6289 ± 0.0363 | 0.5808 ± 0.0529 | 0.5704 ± 0.0506 | 0.6065 ± 0.0237 |
| QwenVisFusion | official_cv_3class_rgb_dual_last2blocks_causal | 3class | causal | rgb_dual | before | 3 | 0.6386 ± 0.0210 | 0.6359 ± 0.0158 | 0.6067 ± 0.0351 | 0.6009 ± 0.0361 | 0.6384 ± 0.0211 |

## Best By Family

| Task | Prompt-Qwen | Classic | QwenVisFusion |
|------|-------------|---------|---------------|
| 2class | `data1_official_cv_2class_baseline` / 0.3471 | `resnet18_triple_view_late_fusion_2class` / 0.5523 | `official_cv_2class_rgb_dual_causal` / 0.4401 |
| 3class | `data1_official_cv_3class_baseline` / 0.4073 | `resnet18_triple_view_late_fusion_3class` / 0.4236 | `official_cv_3class_rgb_dual_causal` / 0.6067 |

## Before vs After

### data1_official_cv_2class_baseline

| Metric | Before | After | Delta(After-Before) |
|--------|--------|-------|---------------------|
| accuracy | 0.4134 | 0.5264 | 0.1130 |
| balanced_accuracy | 0.5000 | 0.4958 | -0.0042 |
| macro_f1 | 0.2925 | 0.3471 | 0.0546 |

### data1_official_cv_3class_baseline

| Metric | Before | After | Delta(After-Before) |
|--------|--------|-------|---------------------|
| accuracy | 0.4134 | 0.4752 | 0.0619 |
| balanced_accuracy | 0.3333 | 0.4586 | 0.1252 |
| macro_f1 | 0.1950 | 0.4073 | 0.2124 |

### data1_official_cv_3class_causal

| Metric | Before | After | Delta(After-Before) |
|--------|--------|-------|---------------------|
| accuracy | 0.4134 | 0.4530 | 0.0396 |
| balanced_accuracy | 0.3333 | 0.4116 | 0.0783 |
| macro_f1 | 0.1950 | 0.3476 | 0.1526 |

### official_cv_3class_rgb_dual_last2blocks_baseline

| Metric | Before | After | Delta(After-Before) |
|--------|--------|-------|---------------------|
| accuracy | 0.6304 | 0.5899 | -0.0404 |
| balanced_accuracy | 0.6303 | 0.6188 | -0.0115 |
| macro_f1 | 0.5964 | 0.5684 | -0.0280 |

### official_cv_3class_rgb_dual_last2blocks_causal

| Metric | Before | After | Delta(After-Before) |
|--------|--------|-------|---------------------|
| accuracy | 0.6386 | 0.6106 | -0.0281 |
| balanced_accuracy | 0.6359 | 0.6289 | -0.0070 |
| macro_f1 | 0.6067 | 0.5808 | -0.0259 |

## Notes

- `data2` is a harder transfer setting than the source-side `official_cv`; its scores should be interpreted as external stress-test results.
- Prompt-Qwen rows use the original prompt-based adapter evaluation pipeline on the new `data2` transfer manifest.
- Classic rows reuse trained `data1 official_cv` checkpoints and only change the evaluation manifest.
- QwenVisFusion rows reuse trained discriminative checkpoints; `last2blocks` runs include both `before` and `after` stages when available.

