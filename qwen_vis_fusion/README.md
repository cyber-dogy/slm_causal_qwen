# Qwen Vision Fusion

这个目录与原始 Qwen SFT 主线解耦，专门放：
- `Qwen2-VL` 视觉编码器
- `rgb_dual` 判别式 `MLP fusion`
- `baseline / causal consistency`

当前主线：
- 输入：`rgb_view1_after + rgb_view2_after`
- backbone：`Qwen2-VL` vision tower
- 头部：`MLP fusion + classifier`
- 因果约束：同标签、不同 condition 的融合表征一致性

常用命令：

```bash
python qwen_vis_fusion/train_cv.py \
  --config qwen_vis_fusion/configs/official_cv/3class_rgb_dual_baseline.json
```

一键跑完整套：

```bash
./qwen_vis_fusion/run_suite.sh
```

跑 `last2 blocks` 真微调并生成 `before/after` 报告：

```bash
./qwen_vis_fusion/run_last2_suite.sh
```

生成报告：

```bash
python qwen_vis_fusion/report.py \
  --run_roots \
    ${PROJECT_OUTPUT_ROOT}/qwen_vis_fusion/official_cv_2class_rgb_dual_baseline \
    ${PROJECT_OUTPUT_ROOT}/qwen_vis_fusion/official_cv_2class_rgb_dual_causal \
    ${PROJECT_OUTPUT_ROOT}/qwen_vis_fusion/official_cv_3class_rgb_dual_baseline \
    ${PROJECT_OUTPUT_ROOT}/qwen_vis_fusion/official_cv_3class_rgb_dual_causal \
  --output_dir ${PROJECT_OUTPUT_ROOT}/qwen_vis_fusion/reports/official_cv
```

后续扩展建议：
- 先比较 `rgb_view1` vs `rgb_dual`
- 再试 `unfreeze_last_n_blocks > 0`
- 再引入 `before/change` 分支
