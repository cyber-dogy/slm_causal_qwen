# Qwen Vision Fusion Report / Qwen 视觉融合报告

## Coverage / 覆盖实验

| Experiment | Task | Method | Input | Split | Folds | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Folded 2Class Macro F1 |
|------------|------|--------|-------|-------|-------|----------|--------------|----------|-------------|------------------------|
| qwen_vis_rgb_dual_baseline_2class | 2class | baseline | rgb_dual | val | 3 | 0.5657 ± 0.0788 | 0.5473 ± 0.0516 | 0.4307 ± 0.0995 | 0.4452 ± 0.1103 | - |
| qwen_vis_rgb_dual_baseline_2class | 2class | baseline | rgb_dual | test | 3 | 0.6676 ± 0.0199 | 0.5381 ± 0.0294 | 0.4650 ± 0.0560 | 0.5574 ± 0.0415 | - |
| qwen_vis_rgb_dual_causal_2class | 2class | causal | rgb_dual | val | 3 | 0.6224 ± 0.1704 | 0.6124 ± 0.1589 | 0.5046 ± 0.2399 | 0.5130 ± 0.2439 | - |
| qwen_vis_rgb_dual_causal_2class | 2class | causal | rgb_dual | test | 3 | 0.6571 ± 0.0267 | 0.5429 ± 0.0606 | 0.4707 ± 0.1145 | 0.5571 ± 0.0840 | - |
| qwen_vis_rgb_dual_baseline_3class | 3class | baseline | rgb_dual | val | 3 | 0.8261 ± 0.0802 | 0.8628 ± 0.0265 | 0.8358 ± 0.0634 | 0.8198 ± 0.0815 | 0.8143 ± 0.0860 |
| qwen_vis_rgb_dual_baseline_3class | 3class | baseline | rgb_dual | test | 3 | 0.8383 ± 0.0824 | 0.8422 ± 0.0866 | 0.8402 ± 0.0856 | 0.8357 ± 0.0834 | 0.8241 ± 0.0808 |
| qwen_vis_rgb_dual_causal_3class | 3class | causal | rgb_dual | val | 3 | 0.8176 ± 0.0894 | 0.8578 ± 0.0300 | 0.8257 ± 0.0752 | 0.8094 ± 0.0926 | 0.8049 ± 0.0948 |
| qwen_vis_rgb_dual_causal_3class | 3class | causal | rgb_dual | test | 3 | 0.8521 ± 0.0758 | 0.8556 ± 0.0794 | 0.8547 ± 0.0780 | 0.8506 ± 0.0761 | 0.8392 ± 0.0754 |

## Baseline vs Causal / Baseline 与 Causal 对比

### 2class / val

| Metric | Baseline | Causal | Delta(Causal-Baseline) |
|--------|----------|--------|------------------------|
| accuracy | 0.5657 | 0.6224 | 0.0567 |
| balanced_accuracy | 0.5473 | 0.6124 | 0.0651 |
| macro_f1 | 0.4307 | 0.5046 | 0.0739 |

### 2class / test

| Metric | Baseline | Causal | Delta(Causal-Baseline) |
|--------|----------|--------|------------------------|
| accuracy | 0.6676 | 0.6571 | -0.0105 |
| balanced_accuracy | 0.5381 | 0.5429 | 0.0048 |
| macro_f1 | 0.4650 | 0.4707 | 0.0058 |

### 3class / val

| Metric | Baseline | Causal | Delta(Causal-Baseline) |
|--------|----------|--------|------------------------|
| accuracy | 0.8261 | 0.8176 | -0.0085 |
| balanced_accuracy | 0.8628 | 0.8578 | -0.0051 |
| macro_f1 | 0.8358 | 0.8257 | -0.0101 |

### 3class / test

| Metric | Baseline | Causal | Delta(Causal-Baseline) |
|--------|----------|--------|------------------------|
| accuracy | 0.8383 | 0.8521 | 0.0139 |
| balanced_accuracy | 0.8422 | 0.8556 | 0.0134 |
| macro_f1 | 0.8402 | 0.8547 | 0.0145 |

## External Comparison / 外部对照

| Task | Split | Current Best QwenVisFusion | Macro F1 | Prompt-Qwen After | Macro F1 | Best Classic | Macro F1 |
|------|-------|---------------------------|----------|-------------------|----------|--------------|----------|
| 2class | val | qwen_vis_rgb_dual_causal_2class (causal) | 0.5046 | 2Class Baseline / 二分类 Baseline | 0.3414 | resnet18_rgb_dual_mlp_concat_2class | 0.8883 |
| 2class | test | qwen_vis_rgb_dual_causal_2class (causal) | 0.4707 | 2Class Baseline / 二分类 Baseline | 0.3903 | resnet18_rgb_dual_mlp_concat_2class | 0.8588 |
| 3class | val | qwen_vis_rgb_dual_baseline_3class (baseline) | 0.8358 | 3Class Baseline / 三分类 Baseline | 0.3794 | mlp_concat | 0.8962 |
| 3class | test | qwen_vis_rgb_dual_causal_3class (causal) | 0.8547 | 3Class Baseline / 三分类 Baseline | 0.3914 | resnet18_rgb_dual_mlp_concat_3class | 0.8323 |

## Notes / 说明

- This branch uses Qwen2-VL as a vision encoder and trains a discriminative MLP fusion head / 该分支将 Qwen2-VL 作为视觉编码器，并训练判别式 MLP 融合头。
- The current official configs freeze the Qwen vision tower and train the fusion head only / 当前 official 配置默认冻结 Qwen 视觉塔，仅训练融合头。
- `baseline` uses classification loss only / `baseline` 仅使用分类损失。
- `causal` adds consistency over fused visual representations for same-label different-condition pairs / `causal` 在融合视觉表征上对同标签不同 condition 正样本加入一致性约束。

