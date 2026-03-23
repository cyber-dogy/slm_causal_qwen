# Experiment Research Log

这个文档是当前项目的长期研究记录，用于后续论文撰写、实验复现和结果回顾。
后续每次新增实验、修正协议、得到新结论，都应继续更新本文件。

## 1. Current Goal

- 任务主线已经从“直接 prompt 分类”转为“稳定的 data1 条件级分类 baseline + 因果一致性验证”。
- 当前最重要问题不是 `data2` 迁移，而是：
  - 在 `data1 official_cv` 上建立可信的 `2class / 3class` 闭集基线
  - 明确 Qwen 路线为什么原始 prompt 微调失败
  - 评估显式视觉特征融合和因果一致性是否能提升稳定性

## 2. Data Protocol

- 数据协议目录：
  `/home/gjw/code/SLM_data/processed_qwen_data1_protocol`
- 正式评估协议：
  `official_cv`
- 划分单位：
  `condition_uid`
- 评估目标：
  - `2class`: `normal / abnormal`
  - `3class`: `normal / HEW / LEL`

相关说明与 split 文档：
- [split_description.md](/home/gjw/code/Qwen-SLM/runs/data1_protocol_reports/official_cv/split_description.md)
- [split_conditions.csv](/home/gjw/code/Qwen-SLM/runs/data1_protocol_reports/official_cv/split_conditions.csv)

## 3. Code Branches

### 3.1 Prompt-Qwen Mainline

- 代码入口：
  [src/train_qwen2vl_slm_lora.py](/home/gjw/code/Qwen-SLM/src/train_qwen2vl_slm_lora.py)
- 特点：
  - Qwen2-VL + prompt-based classification
  - 使用候选标签打分
  - 原始 causal 一致性加在 token hidden state 上

### 3.2 Classic Multimodal Baselines

- 独立目录：
  [classic_mm_baselines](/home/gjw/code/Qwen-SLM/classic_mm_baselines/README.md)
- 当前已实现：
  - `ResNet18 + rgb_view1`
  - `rgb_dual + mlp_concat`
  - `cross_attention`
  - `late_fusion`
  - `ir_only`

总报告：
- [official_cv_extended comparison_report.md](/home/gjw/code/Qwen-SLM/runs/classic_mm_baselines/reports/official_cv_extended/comparison_report.md)

### 3.3 Qwen Vision Fusion

- 独立目录：
  [qwen_vis_fusion](/home/gjw/code/Qwen-SLM/qwen_vis_fusion/README.md)
- 当前路线：
  - Qwen2-VL vision tower 作为显式视觉编码器
  - `rgb_dual`
  - `MLP fusion`
  - 因果一致性约束直接加在融合视觉表征上

总报告：
- [official_cv comparison_report.md](/home/gjw/code/Qwen-SLM/runs/qwen_vis_fusion/reports/official_cv/comparison_report.md)
- `last2 blocks` 真微调报告：
  [official_cv_last2 comparison_report.md](/home/gjw/code/Qwen-SLM/runs/qwen_vis_fusion/reports/official_cv_last2/comparison_report.md)

## 4. Main Findings So Far

### 4.1 Prompt-Qwen 路线

- 原始 prompt-Qwen 在 `data1 official_cv` 上效果不稳定。
- 修复评估后，结论仍然是：
  - `2class` 几乎无效
  - `3class` 有一定增益，但明显落后于经典判别式视觉模型
- 关键原因：
  - 没有显式视觉融合头
  - 仍然是语言建模式闭集分类
  - causal 约束位置不对，加在 token hidden state 而不是视觉表征上

参考报告：
- [data1 official_cv comparison_report.md](/home/gjw/code/Qwen-SLM/runs/data1_protocol_reports/official_cv/comparison_report.md)

### 4.2 Classic Baselines

当前最强经典模型结论：
- `2class` 最强：
  `ResNet18 + rgb_dual + mlp_concat`
  - test `macro_f1 = 0.8588`
- `3class` 最强：
  `ResNet18 + rgb_dual + mlp_concat`
  - test `macro_f1 = 0.8323`

这说明：
- `rgb_dual` 是当前协议下很强的输入组合
- `LEL` 并不是“完全不可学”
- 简单判别式融合头非常有效

### 4.3 Qwen Vision Fusion Frozen-Head

当前最强 QwenVision + fusion 结果：
- `2class causal`
  - test `macro_f1 = 0.4707`
- `3class baseline`
  - test `macro_f1 = 0.8402`
- `3class causal`
  - test `macro_f1 = 0.8547`

这说明：
- Qwen 作为视觉编码器 + 显式融合头后，性能发生质变
- `3class` 路线已经略高于当前经典最强 `0.8323`
- 把 causal 一致性加在视觉融合表征上，是有效方向

参考报告：
- [qwen_vis_fusion comparison_report.md](/home/gjw/code/Qwen-SLM/runs/qwen_vis_fusion/reports/official_cv/comparison_report.md)

### 4.3.1 QwenVis 输入消融补充

为支撑论文中“工业大模型多模态融合架构”的论证，本轮补充了 `QwenVisFusion 3class` 的输入消融：

- `rgb_view1 baseline`
  - test `macro_f1 = 0.7419`
- `ir_only baseline`
  - test `macro_f1 = 0.6270`
- `rgb_dual baseline`
  - test `macro_f1 = 0.8402`
- `triple_view baseline`
  - test `macro_f1 = 0.8121`
- `rgb_dual causal`
  - test `macro_f1 = 0.8547`

结论：
- 显式双 RGB 融合明显优于单模态输入。
- 当前架构下，直接把 IR 与双 RGB 一起并入，并没有超过 `rgb_dual`。
- 因此论文里更准确的表述应当是：
  “本文提出了适用于工业小样本场景的 Qwen 显式融合架构，并通过输入消融证明双 RGB 融合是当前任务下的最优配置；在此基础上引入表征级因果一致性约束后得到最终最佳结果。”

### 4.4 Qwen Vision Fusion Last2 True Fine-Tuning

在 frozen-head `3class` 强基线之上，进一步解冻最后 `2` 个视觉 block 做真微调后，结果没有继续提升，反而出现明显退化。

`3class baseline`：
- `before` = frozen-head checkpoint
  - test `macro_f1 = 0.8402`
- `after` = 解冻最后 2 个视觉 block 真微调后
  - test `macro_f1 = 0.7365`
- `delta = -0.1037`

`3class causal`：
- `before` = frozen-head causal checkpoint
  - test `macro_f1 = 0.8547`
- `after` = 解冻最后 2 个视觉 block 真微调后
  - test `macro_f1 = 0.7781`
- `delta = -0.0766`

这说明：
- 当前 `last2 blocks` 真微调存在明显灾难性遗忘/表征漂移风险
- causal 一致性在真微调时有一定缓冲作用，但仍不足以超过 frozen-head 版本
- 现阶段最稳的主结果仍然是：
  `QwenVisFusion + rgb_dual + frozen vision tower + causal consistency`

参考报告：
- [official_cv_last2 comparison_report.md](/home/gjw/code/Qwen-SLM/runs/qwen_vis_fusion/reports/official_cv_last2/comparison_report.md)

## 5. Current Best Results Snapshot

| Family | Task | Model | Test Macro F1 | Notes |
|--------|------|-------|---------------|-------|
| Prompt-Qwen | 2class | prompt baseline | 0.3903 | 基本无效 |
| Prompt-Qwen | 3class | prompt baseline | 0.3914 | 显著落后 |
| Classic | 2class | ResNet18 + rgb_dual + mlp_concat | 0.8588 | 当前经典最强 |
| Classic | 3class | ResNet18 + rgb_dual + mlp_concat | 0.8323 | 当前经典最强 |
| QwenVisFusion | 2class | rgb_dual causal | 0.4707 | 仍偏弱 |
| QwenVisFusion | 3class | rgb_dual causal | 0.8547 | 当前总体最好 |
| QwenVisFusion Last2 FT | 3class | rgb_dual last2 causal | 0.7781 | 真微调后退化 |

## 6. Last2 Visual Blocks True Fine-Tuning

目标与实现：
- 在 `QwenVisFusion 3class` 强基线之上
- 解冻最后 `2` 个视觉 block
- 做真正的视觉微调
- 报告显式展示：
  - `before / after`
  - frozen baseline vs true fine-tune
  - causal vs non-causal
- 代码已接入：
  - `init_run_root` 加载 frozen checkpoint
  - `eval_before_training`
  - last2 true fine-tune configs
  - W&B
  - `before/after` 报告逻辑

相关脚本：
- [run_last2_suite.sh](/home/gjw/code/Qwen-SLM/qwen_vis_fusion/run_last2_suite.sh)

本轮关键修正：
- 初版 last2 配置出现了明显灾难性遗忘
- 后续修正包括：
  - 增加 `before/after` 评估
  - 视觉 block 与融合头分离学习率
  - 梯度裁剪
  - 解冻 block 改为 `fp32` 真训练
  - mixed-dtype 评估修复

正式结果：
- `unfrozen baseline`
  - test `macro_f1 = 0.7365`
  - 相比 frozen baseline `0.8402`，下降 `0.1037`
- `unfrozen causal`
  - test `macro_f1 = 0.7781`
  - 相比 frozen causal `0.8547`，下降 `0.0766`

结论：
- 当前 true fine-tune 还不是主线最优解
- 若继续做视觉真微调，需要更保守的策略：
  - 更低 visual LR
  - 更短 schedule
  - 更强冻结策略或逐层解冻
  - adapter/LoRA 式视觉微调，替代直接解冻 block

## 7. Writing Notes

论文里目前可以明确写出的核心论点：
- 直接 prompt 微调的 Qwen2-VL 并不适合当前小样本 condition-heldout 缺陷分类。
- 显式视觉编码 + 显式融合头，是把 Qwen 用在该任务上的关键转折。
- 因果一致性若加在 token-level hidden states 上帮助有限；加在视觉融合表征上更有效。
- `3class` 学习再折叠到异常识别，可能比直接 `2class` 更稳。
- 当前视觉塔“解冻最后 2 个 block”真微调并未优于冻结视觉塔，说明在小样本协议下，稳定表征比激进适配更重要。

## 8. Latest Update (2026-03-23)

本次新增：
- `QwenVisFusion last2 blocks` 真微调正式实验
- `before / after` 对比报告
- frozen vs unfrozen 对比
- 长期研究总结文档持续维护

新增报告：
- [official_cv_last2 comparison_report.md](/home/gjw/code/Qwen-SLM/runs/qwen_vis_fusion/reports/official_cv_last2/comparison_report.md)

本次最重要结论：
- 当前最佳 Qwen 路线仍然是：
  `QwenVisFusion + rgb_dual + frozen vision tower + causal consistency`
- 真微调虽已具备完整前后对比和 W&B 记录，但结果显示性能退化，不宜直接作为主结果写入论文。

## 9. Data2 External Transfer Stress Test (2026-03-23)

### 9.1 Protocol Positioning

- `data2` 不再被视为简单跨工况测试，而是更困难的外部迁移场景：
  - 未见模型
  - 未见工艺
  - 角色/纹理/参数复合分布偏移
- 因此 `data2` 结果只用于 external transfer stress test，不参与主选模。
- 本轮为避免污染既有主线代码，新建了独立目录：
  [data2_transfer_eval](/home/gjw/code/Qwen-SLM/data2_transfer_eval/README.md)

协议与数据分析文件：
- [data2_analysis.md](/home/gjw/code/SLM_data/processed_qwen_data2_transfer/data2_analysis.md)
- [transfer_test_manifest.jsonl](/home/gjw/code/SLM_data/processed_qwen_data2_transfer/transfer_test_manifest.jsonl)
- [split_summary.json](/home/gjw/code/SLM_data/processed_qwen_data2_transfer/split_summary.json)

### 9.2 Data2 Overview

- `data1`:
  - samples = `289`
  - conditions = `22`
  - labels = `normal 104 / HEW 86 / LEL 99`
- `data2`:
  - samples = `404`
  - conditions = `22`
  - labels = `normal 167 / HEW 128 / LEL 109`
- `data2` 角色构成明显更复杂：
  - `causal_intervention_low = 82`
  - `unseen_texture_shift = 76`
  - `causal_intervention_high = 66`
  - `causal_equiv_test = 41`
  - `unseen_strategy_shift = 40`
  - `unseen_composite_shift = 35`
  - `unseen_param_shift = 28`
  - `source_repeat = 20`
  - `source_train = 16`

### 9.3 Transfer Evaluation Coverage

统一 transfer 报告：
- [comparison_report.md](/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval/reports/comparison_report.md)
- [comparison_report.zh_en.md](/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval/reports/comparison_report.zh_en.md)

参与迁移评估的模型家族：
- Prompt-Qwen
- Classic multimodal baselines
- QwenVisFusion
- QwenVisFusion `last2 blocks` true fine-tuning

### 9.4 Main Transfer Findings

`2class` 最优家族对比：
- Prompt-Qwen 最佳：
  `data1_official_cv_2class_baseline`
  - transfer `macro_f1 = 0.3471`
  - `balanced_accuracy = 0.4958`
- Classic 最佳：
  `ResNet18 + triple_view + late_fusion`
  - transfer `macro_f1 = 0.5523`
  - `balanced_accuracy = 0.5909`
- QwenVisFusion 最佳：
  `official_cv_2class_rgb_dual_causal`
  - transfer `macro_f1 = 0.4401`
  - `balanced_accuracy = 0.5289`

`3class` 最优家族对比：
- Prompt-Qwen 最佳：
  `data1_official_cv_3class_baseline`
  - transfer `macro_f1 = 0.4073`
  - `balanced_accuracy = 0.4586`
- Classic 最佳：
  `ResNet18 + triple_view + late_fusion`
  - transfer `macro_f1 = 0.4236`
  - `balanced_accuracy = 0.4973`
- QwenVisFusion 最佳：
  `official_cv_3class_rgb_dual_causal`
  - transfer `macro_f1 = 0.6067`
  - `balanced_accuracy = 0.6359`

### 9.5 Before/After Transfer Comparison

Prompt-Qwen:
- `2class baseline`
  - `before macro_f1 = 0.2925`
  - `after macro_f1 = 0.3471`
  - 有小幅提升，但仍接近塌缩边界
- `3class baseline`
  - `before macro_f1 = 0.1950`
  - `after macro_f1 = 0.4073`
  - 明显优于 base model，但仍落后于更强判别式模型
- `3class causal`
  - `before macro_f1 = 0.1950`
  - `after macro_f1 = 0.3476`
  - 低于 prompt baseline

QwenVisFusion `last2 blocks` true fine-tuning:
- `3class baseline`
  - `before macro_f1 = 0.5964`
  - `after macro_f1 = 0.5684`
  - 解冻后下降 `0.0280`
- `3class causal`
  - `before macro_f1 = 0.6067`
  - `after macro_f1 = 0.5808`
  - 解冻后下降 `0.0259`

### 9.6 Interpretation

- `data2` 上最稳的路线仍然是：
  `QwenVisFusion + rgb_dual + frozen vision tower + causal consistency`
- Prompt-Qwen 在 harder transfer 场景中依然显著落后，说明“prompt-based label scoring”不是当前任务的最佳范式。
- 经典模型在 `data2` 上仍有竞争力，但最好结果低于 `QwenVisFusion 3class causal`，说明显式 Qwen 视觉表征在更强外部分布偏移下更有潜力。
- `last2 blocks` 真微调在 `data1 official_cv` 和 `data2 transfer` 上都没有带来收益，现阶段应视为不稳定方案。
- 当前最适合作为论文主线的结论组合是：
  - `data1 official_cv` 作为主闭集验证
  - `data2 transfer` 作为外部 stress test
  - 主模型使用 `QwenVisFusion 3class causal`

## 10. Repository Packaging And Demo

### 10.1 GitHub-Ready Paper Project

新增独立目录：
- `slm_causal_qwen_paper_project/`

目标：
- 不污染当前主工作区
- 汇总论文主线相关代码、文档和图表脚本
- 默认不包含 `outputs/`、`wandb/`、数据集和模型权重
- 便于后续直接整理成公开仓库

当前已整理内容：
- `qwen_vis_fusion/`
- `classic_mm_baselines/`
- `data2_transfer_eval/`
- `paper_visualization/`
- `demo/`
- `docs/`
- `scripts/`
- `src/`

新增工程级文件：
- `README.md`
- `.gitignore`
- `requirements.txt`
- `configs/local_paths.example.env`
- `configs/demo.example.yaml`

### 10.2 Path And Reproducibility Cleanup

本轮对独立工程做了以下收口：
- `qwen_vis_fusion` 与 `classic_mm_baselines` 的训练脚本支持 `${PROJECT_ROOT}`、`${SLM_DATA1_PROTOCOL_ROOT}`、`${QWEN2_VL_MODEL_PATH}` 这类环境变量占位符
- 一键脚本改成相对仓库路径，不再硬编码 `/home/gjw/...`
- `paper_visualization` 改成默认读取 `${PROJECT_OUTPUT_ROOT}` 下的结果
- notebook 重新生成为相对路径版本

这意味着后续迁到新机器时，优先只需要：
- 复制 `configs/local_paths.example.env`
- 填本地数据与模型路径
- 把实验输出放到 `outputs/`

### 10.3 Gradio Comparison Demo

新增目录：
- `demo/`

当前 demo 设计：
- 左侧：专用 `QwenVisFusion + rgb_dual + causal consistency`
- 右侧：通用 `Qwen2-VL` 单图 VQA
- 输入：`rgb_view1`、`rgb_view2`、可选 `ir`
- 右侧通用模型先把多张图拼成 montage，再做单图问答

当前 smoke test 已通过：
- demo UI 可正常构建
- 专用模型可从 `official_cv_3class_rgb_dual_causal` 自动选择 best fold
- 使用真实样本测试时，专用模型成功输出：
  - `pred_label = normal`
  - `confidence = 0.8977`
  - `fold = fold_02`
- 通用模型在同一 montage 上成功返回自然语言回答

这一步的意义：
- 方便和此前通用工业 VQA demo 做同屏对照
- 也方便后续论文答辩或投稿补充材料展示

## 11. Update Policy

后续每次实验更新本文件时，至少追加：
- 实验日期
- 协议和输入
- 改动内容
- 主要指标
- 结论
- 下一步计划
