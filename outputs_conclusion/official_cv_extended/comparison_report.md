# Classic Multimodal Baseline Report / 经典多模态基线报告

## Coverage / 覆盖实验

| Experiment | Task | Backbone | Input | Fusion | Split | Folds | Accuracy | Balanced Acc | Macro F1 | Weighted F1 |
|------------|------|----------|-------|--------|-------|-------|----------|--------------|----------|-------------|
| single_view | 2class | resnet18 | triple_view | single_view | val | 3 | 0.7749 ± 0.0919 | 0.7955 ± 0.0779 | 0.7714 ± 0.0946 | 0.7686 ± 0.0977 |
| single_view | 2class | resnet18 | triple_view | single_view | test | 3 | 0.7755 ± 0.0290 | 0.7522 ± 0.0571 | 0.7485 ± 0.0435 | 0.7703 ± 0.0339 |
| mlp_concat | 2class | resnet18 | triple_view | mlp_concat | val | 3 | 0.8856 ± 0.0607 | 0.8825 ± 0.0676 | 0.8824 ± 0.0634 | 0.8846 ± 0.0619 |
| mlp_concat | 2class | resnet18 | triple_view | mlp_concat | test | 3 | 0.8344 ± 0.0426 | 0.8247 ± 0.0647 | 0.8192 ± 0.0537 | 0.8336 ± 0.0459 |
| cross_attention | 2class | resnet18 | triple_view | cross_attention | val | 3 | 0.8727 ± 0.0988 | 0.8798 ± 0.0912 | 0.8725 ± 0.0986 | 0.8723 ± 0.0992 |
| cross_attention | 2class | resnet18 | triple_view | cross_attention | test | 3 | 0.8347 ± 0.0713 | 0.8063 ± 0.0879 | 0.8124 ± 0.0852 | 0.8305 ± 0.0750 |
| resnet18_ir_only_single_view_2class | 2class | resnet18 | ir_only | single_view | val | 3 | 0.7514 ± 0.1695 | 0.7580 ± 0.1420 | 0.7287 ± 0.1901 | 0.7247 ± 0.2019 |
| resnet18_ir_only_single_view_2class | 2class | resnet18 | ir_only | single_view | test | 3 | 0.7547 ± 0.0933 | 0.6916 ± 0.1199 | 0.6928 ± 0.1284 | 0.7309 ± 0.1094 |
| resnet18_rgb_dual_mlp_concat_2class | 2class | resnet18 | rgb_dual | mlp_concat | val | 3 | 0.8906 ± 0.1046 | 0.8976 ± 0.0898 | 0.8883 ± 0.1061 | 0.8876 ± 0.1085 |
| resnet18_rgb_dual_mlp_concat_2class | 2class | resnet18 | rgb_dual | mlp_concat | test | 3 | 0.8730 ± 0.0910 | 0.8506 ± 0.0981 | 0.8588 ± 0.1011 | 0.8715 ± 0.0919 |
| resnet18_triple_view_late_fusion_2class | 2class | resnet18 | triple_view | late_fusion | val | 3 | 0.8800 ± 0.0534 | 0.8768 ± 0.0514 | 0.8778 ± 0.0539 | 0.8796 ± 0.0534 |
| resnet18_triple_view_late_fusion_2class | 2class | resnet18 | triple_view | late_fusion | test | 3 | 0.8173 ± 0.1003 | 0.7965 ± 0.1160 | 0.7974 ± 0.1152 | 0.8149 ± 0.1034 |
| single_view | 3class | resnet18 | triple_view | single_view | val | 3 | 0.8505 ± 0.0763 | 0.8805 ± 0.0492 | 0.8549 ± 0.0749 | 0.8507 ± 0.0775 |
| single_view | 3class | resnet18 | triple_view | single_view | test | 3 | 0.8237 ± 0.0440 | 0.8348 ± 0.0429 | 0.8269 ± 0.0483 | 0.8183 ± 0.0482 |
| mlp_concat | 3class | resnet18 | triple_view | mlp_concat | val | 3 | 0.8902 ± 0.0865 | 0.9015 ± 0.0580 | 0.8962 ± 0.0727 | 0.8912 ± 0.0829 |
| mlp_concat | 3class | resnet18 | triple_view | mlp_concat | test | 3 | 0.8138 ± 0.1127 | 0.8135 ± 0.1131 | 0.8106 ± 0.1143 | 0.8091 ± 0.1161 |
| cross_attention | 3class | resnet18 | triple_view | cross_attention | val | 3 | 0.8676 ± 0.1060 | 0.8838 ± 0.0653 | 0.8803 ± 0.0829 | 0.8695 ± 0.1005 |
| cross_attention | 3class | resnet18 | triple_view | cross_attention | test | 3 | 0.8098 ± 0.0082 | 0.8198 ± 0.0117 | 0.8134 ± 0.0101 | 0.8045 ± 0.0096 |
| resnet18_ir_only_single_view_3class | 3class | resnet18 | ir_only | single_view | val | 3 | 0.8518 ± 0.1138 | 0.8658 ± 0.0854 | 0.8469 ± 0.1132 | 0.8483 ± 0.1162 |
| resnet18_ir_only_single_view_3class | 3class | resnet18 | ir_only | single_view | test | 3 | 0.7271 ± 0.1127 | 0.7298 ± 0.1160 | 0.7117 ± 0.1254 | 0.7085 ± 0.1262 |
| resnet18_rgb_dual_mlp_concat_3class | 3class | resnet18 | rgb_dual | mlp_concat | val | 3 | 0.8808 ± 0.0806 | 0.9101 ± 0.0329 | 0.8912 ± 0.0672 | 0.8804 ± 0.0798 |
| resnet18_rgb_dual_mlp_concat_3class | 3class | resnet18 | rgb_dual | mlp_concat | test | 3 | 0.8349 ± 0.0858 | 0.8412 ± 0.0846 | 0.8323 ± 0.0892 | 0.8293 ± 0.0917 |
| resnet18_triple_view_late_fusion_3class | 3class | resnet18 | triple_view | late_fusion | val | 3 | 0.8139 ± 0.1142 | 0.8218 ± 0.1119 | 0.8048 ± 0.1180 | 0.8059 ± 0.1174 |
| resnet18_triple_view_late_fusion_3class | 3class | resnet18 | triple_view | late_fusion | test | 3 | 0.7747 ± 0.0503 | 0.7786 ± 0.0531 | 0.7751 ± 0.0522 | 0.7707 ± 0.0556 |

## Notes / 说明

- Input modes are read from the same data1 manifests and only differ in selected after-image views / 输入模式来自同一份 data1 manifest，只是选择的 after 视图不同。
- `rgb_view1` means only the first RGB after-view / `rgb_view1` 表示仅使用第一路 RGB after 图像。
- `single_view` always uses the first selected view, so `triple_view + single_view` is effectively `rgb_view1_after` / `single_view` 总是取当前输入模式的第一路视图，因此 `triple_view + single_view` 实际等价于只用 `rgb_view1_after`。
- `rgb_dual` means `rgb_view1_after + rgb_view2_after` / `rgb_dual` 表示双 RGB after 输入。
- `ir_only` means only `ir_after` / `ir_only` 表示仅使用 `ir_after`。
- `late_fusion` performs per-view classification before weighted logit fusion / `late_fusion` 先做单视图分类，再做加权 logits 融合。

