# Classic Multimodal Baselines

这个目录与 Qwen 主线解耦，专门放经典视觉模型 baseline。

当前实现：
- `ResNet18 + single_view`
- `ResNet18 + mean_pool`
- `ResNet18 + mlp_concat`
- `ResNet18 + gated_mlp`
- `ResNet18 + cross_attention`
- `ResNet18 + late_fusion`
- config-driven `backbone/input/fusion` experiments for future CNN/Transformer extensions

输入默认与当前 Qwen v1 保持一致：
- `rgb_view1_after`
- `rgb_view2_after`
- `ir_after`

默认协议：
- 使用 `${SLM_DATA1_PROTOCOL_ROOT}/official_cv`
- `train` 直接复用 manifest 中的上采样结果
- `val/test` 保持自然分布

常用命令：

```bash
./classic_mm_baselines/run_suite.sh
```

单个实验：

```bash
python classic_mm_baselines/train_cv.py \
  --task_mode 3class \
  --backbone_name resnet18 \
  --input_mode triple_view \
  --fusion_type mlp_concat \
  --protocol_root ${SLM_DATA1_PROTOCOL_ROOT} \
  --run_root ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_mlp_concat \
  --fold all
```

配置方式运行：

```bash
python classic_mm_baselines/train_cv.py \
  --config classic_mm_baselines/configs/official_cv/3class_triple_view_late_fusion.json
```

新增消融一键运行：

```bash
./classic_mm_baselines/run_ablation_suite.sh
```

生成报告：

```bash
python classic_mm_baselines/report.py \
  --run_roots \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_single_view \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_mlp_concat \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_cross_attention \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_ir_only_single_view \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_rgb_dual_mlp_concat \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_2class_triple_view_late_fusion \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_single_view \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_mlp_concat \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_cross_attention \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_ir_only_single_view \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_rgb_dual_mlp_concat \
    ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/official_cv_3class_triple_view_late_fusion \
  --output_dir ${PROJECT_OUTPUT_ROOT}/classic_mm_baselines/reports/official_cv_extended
```

常用输入模式：
- `rgb_view1`
- `rgb_dual`
- `ir_only`
- `triple_view`

当前 backbone registry：
- `resnet18`
- `resnet34`
- `resnet50`
- `efficientnet_b0`
- `vit_b_16`

如何继续扩更多经典 baseline：

1. 复制配置模板：

```bash
cp classic_mm_baselines/configs/templates/experiment_template.json \
   classic_mm_baselines/configs/official_cv/3class_resnet50_rgb_dual_mean_pool.json
```

2. 修改这些关键字段：
- `task_mode`: `2class` or `3class`
- `backbone_name`: `resnet18/resnet34/resnet50/efficientnet_b0/vit_b_16`
- `input_mode`: `rgb_view1/rgb_view2/rgb_dual/ir_only/triple_view`
- `fusion_type`: `single_view/mean_pool/mlp_concat/gated_mlp/cross_attention/late_fusion`
- `run_root`: 指向新的实验目录

3. 直接运行：

```bash
python classic_mm_baselines/train_cv.py \
  --config classic_mm_baselines/configs/official_cv/3class_resnet50_rgb_dual_mean_pool.json
```

4. 训练完成后，把新的 `run_root` 加到报告命令里即可。

现成示例配置：
- `classic_mm_baselines/configs/examples/3class_resnet50_rgb_dual_mean_pool.json`
- `classic_mm_baselines/configs/examples/3class_vit_b16_rgb_dual_mean_pool.json`

扩 backbone 的最小改动位置：
- backbone registry: `classic_mm_baselines/encoders.py`
- fusion registry: `classic_mm_baselines/models.py`
- 训练配置入口: `classic_mm_baselines/train_cv.py`
