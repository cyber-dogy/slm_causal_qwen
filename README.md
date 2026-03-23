# SLM Causal Qwen Paper Project

面向论文复现与展示的独立工程目录。这个仓库是从主实验工作区中整理出来的“干净分支”，目标是：

- 保留论文主线相关代码
- 不上传本地产生的训练记录、数据集和大模型权重
- 提供可复现的训练/评估入口
- 提供论文图表整理脚本
- 提供一个可对比展示的 Gradio demo

## 对应论文主线

当前论文主线聚焦：

1. 工业大模型在小样本 SLM 缺陷识别场景下的显式多模态融合架构
2. 表征级因果一致性约束的判别式多模态缺陷识别框架
3. 证明有效迁移的关键不在 prompt 微调，而在稳定视觉编码与显式判别式适配

当前最优实验组是：

- `QwenVisFusion + rgb_dual + causal consistency`
- 主任务：`data1 official_cv / 3class`

## 仓库结构

- [`qwen_vis_fusion/`](qwen_vis_fusion): 论文主方法
- [`classic_mm_baselines/`](classic_mm_baselines): 经典 CNN/Transformer 多模态 baseline
- [`data2_transfer_eval/`](data2_transfer_eval): `data2` 外部迁移评估
- [`paper_visualization/`](paper_visualization): 论文图表生成
- [`demo/`](demo): 专用模型 vs 通用模型对比展示
- [`docs/`](docs): 论文草稿和实验研究日志
- [`scripts/`](scripts): 保留下来的 prompt-Qwen / 数据协议脚本
- [`src/`](src): 原始 prompt-Qwen 主线核心代码

## 不包含的内容

以下内容默认不上传：

- 数据集原图和 manifest 产物
- 模型参数、checkpoint、adapter
- `wandb/`
- `outputs/`
- `paper_visualization/cache/`
- `paper_visualization/exports/`

这些内容都已经被 [`.gitignore`](.gitignore) 排除了。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置本地路径

复制环境变量模板：

```bash
cp configs/local_paths.example.env configs/local_paths.local.env
source configs/local_paths.local.env
```

至少需要配置：

- `SLM_DATA1_PROTOCOL_ROOT`
- `SLM_DATA2_TRANSFER_ROOT`
- `QWEN2_VL_MODEL_PATH`
- `PROJECT_OUTPUT_ROOT`

### 3. 训练论文主方法

```bash
python qwen_vis_fusion/train_cv.py \
  --config qwen_vis_fusion/configs/official_cv/3class_rgb_dual_causal.json
```

### 4. 训练经典 baseline

```bash
python classic_mm_baselines/train_cv.py \
  --config classic_mm_baselines/configs/official_cv/3class_rgb_dual_mlp_concat.json
```

### 5. 生成论文图表

```bash
python paper_visualization/prepare_paper_data.py
python paper_visualization/plot_paper_figures.py
python paper_visualization/make_notebook.py
```

## Demo

对比展示入口在：

- [`demo/app.py`](demo/app.py)

它会把：

- 左侧的专用 `QwenVisFusion` 最佳模型
- 和右侧的通用 `Qwen2-VL` VQA

放在同一个页面里展示，便于和你之前的工业通用 demo 做效果对比。

启动方式：

```bash
python demo/app.py --config configs/demo.local.yaml
```

详细说明见：

- [`demo/README.md`](demo/README.md)

## 论文与实验记录

- 论文草稿：[`docs/paper_draft.md`](docs/paper_draft.md)
- 研究日志：[`docs/experiment_research_log.md`](docs/experiment_research_log.md)

## 说明

- 这个独立工程目录尽量保持可复现，但 `docs/` 中的历史记录和草稿仍然保留了部分原始工作区引用，便于回溯实验上下文。
- 主训练与图表脚本已经改成优先使用相对路径和环境变量，不再强依赖当前机器的绝对目录。
