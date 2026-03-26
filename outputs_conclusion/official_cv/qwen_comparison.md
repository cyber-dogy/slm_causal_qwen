# Qwen vs Classic Baselines / Qwen 与经典基线对照

同一 `data1 official_cv` 条件划分，对照 Qwen after 与经典 ResNet18 基线。

| Metric | Qwen 3/2class | single_view | mlp_concat | cross_attention |
|--------|---------------|-------------|------------|-----------------|
| 2class test macro_f1 | 0.3903 | 0.7485 | 0.8192 | 0.8124 |
| 2class test balanced_accuracy | 0.5000 | 0.7522 | 0.8247 | 0.8063 |
| 3class test macro_f1 | 0.3914 | 0.8269 | 0.8106 | 0.8134 |
| 3class test balanced_accuracy | 0.5021 | 0.8348 | 0.8135 | 0.8198 |

## Key Takeaways / 关键结论

- `2class` 上经典视觉基线显著优于当前 Qwen 2class after。
- `3class` 上经典视觉基线也显著优于当前 Qwen 3class after。
- `3class single_view` 已经很强，说明当前输入下 LEL 并非完全不可学。
- 三图融合对 `2class` 帮助更明显；对 `3class` 没有稳定超过 `single_view`。
