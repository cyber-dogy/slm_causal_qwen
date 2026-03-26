# Classic Multimodal Baseline Report / 经典多模态基线报告

## Coverage / 覆盖实验

| Experiment | Task | Split | Folds | Accuracy | Balanced Acc | Macro F1 | Weighted F1 |
|------------|------|-------|-------|----------|--------------|----------|-------------|
| single_view | 2class | val | 3 | 0.7749 ± 0.0919 | 0.7955 ± 0.0779 | 0.7714 ± 0.0946 | 0.7686 ± 0.0977 |
| single_view | 2class | test | 3 | 0.7755 ± 0.0290 | 0.7522 ± 0.0571 | 0.7485 ± 0.0435 | 0.7703 ± 0.0339 |
| mlp_concat | 2class | val | 3 | 0.8856 ± 0.0607 | 0.8825 ± 0.0676 | 0.8824 ± 0.0634 | 0.8846 ± 0.0619 |
| mlp_concat | 2class | test | 3 | 0.8344 ± 0.0426 | 0.8247 ± 0.0647 | 0.8192 ± 0.0537 | 0.8336 ± 0.0459 |
| cross_attention | 2class | val | 3 | 0.8727 ± 0.0988 | 0.8798 ± 0.0912 | 0.8725 ± 0.0986 | 0.8723 ± 0.0992 |
| cross_attention | 2class | test | 3 | 0.8347 ± 0.0713 | 0.8063 ± 0.0879 | 0.8124 ± 0.0852 | 0.8305 ± 0.0750 |
| single_view | 3class | val | 3 | 0.8505 ± 0.0763 | 0.8805 ± 0.0492 | 0.8549 ± 0.0749 | 0.8507 ± 0.0775 |
| single_view | 3class | test | 3 | 0.8237 ± 0.0440 | 0.8348 ± 0.0429 | 0.8269 ± 0.0483 | 0.8183 ± 0.0482 |
| mlp_concat | 3class | val | 3 | 0.8902 ± 0.0865 | 0.9015 ± 0.0580 | 0.8962 ± 0.0727 | 0.8912 ± 0.0829 |
| mlp_concat | 3class | test | 3 | 0.8138 ± 0.1127 | 0.8135 ± 0.1131 | 0.8106 ± 0.1143 | 0.8091 ± 0.1161 |
| cross_attention | 3class | val | 3 | 0.8676 ± 0.1060 | 0.8838 ± 0.0653 | 0.8803 ± 0.0829 | 0.8695 ± 0.1005 |
| cross_attention | 3class | test | 3 | 0.8098 ± 0.0082 | 0.8198 ± 0.0117 | 0.8134 ± 0.0101 | 0.8045 ± 0.0096 |

## Notes / 说明

- Input follows the current Qwen v1 setup: `rgb_view1_after + rgb_view2_after + ir_after` / 输入与当前 Qwen v1 一致。
- Train manifests reuse the same `official_cv` condition split and manifest-level upsampling / 训练清单复用相同 `official_cv` condition 划分与 manifest 级上采样。
- These numbers are intended as classical-model baselines against the repaired Qwen official report / 这些结果用于和修复后的 Qwen official report 做基线比较。

