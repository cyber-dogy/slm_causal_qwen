# SLM Causal Qwen Paper Project

面向论文复现与展示的独立工程目录。这个仓库是从主实验工作区中整理出来的“干净分支”，目标是：

- 保留论文主线相关代码
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

## 最小复现流程

如果你只想把论文主线从零跑通，按下面 5 步即可：

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置本地环境变量

```bash
cp configs/local_paths.example.env configs/local_paths.local.env
source configs/local_paths.local.env
```

3. 训练主方法

```bash
python qwen_vis_fusion/train_cv.py \
  --config qwen_vis_fusion/configs/official_cv/3class_rgb_dual_causal.json
```

4. 生成主报告与论文图表

```bash
python paper_visualization/prepare_paper_data.py
python paper_visualization/plot_paper_figures.py
```

5. 启动对比 demo

```bash
cp configs/demo.example.yaml configs/demo.local.yaml
python demo/app.py --config configs/demo.local.yaml
```

如果你已经有现成权重，可以跳过第 3 步，直接配置 `specialized.run_root`。

## 模型权重获取方式

这个工程默认不包含任何模型权重。当前支持两种方式：

### 方式 A：自行训练

这是最推荐的复现路径。训练完成后，把 demo 配置中的：

- `specialized.run_root`

指向你自己的实验输出目录即可。主方法最关键的训练入口是：

```bash
python qwen_vis_fusion/train_cv.py \
  --config qwen_vis_fusion/configs/official_cv/3class_rgb_dual_causal.json
```

经典 baseline 可按需训练：

```bash
python classic_mm_baselines/train_cv.py \
  --config classic_mm_baselines/configs/official_cv/3class_rgb_dual_mlp_concat.json
```

### 方式 B：使用后续发布的预训练权重

后续会补充 Quark 网盘分享信息。下载并解压后，只需要把：

- `specialized.run_root`

改成解压后的运行目录即可。

详细说明见：

- [docs/model_weights.md](docs/model_weights.md)

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
- [`docs/model_weights.md`](docs/model_weights.md)

## 发布前检查

这个独立工程默认适合作为 GitHub 论文工程上传，但正式上传前仍建议做一次检查。

建议优先执行：

```bash
bash scripts/check_release_tree.sh
```

这会帮助你检查：

- 本地配置文件是否还留在目录里
- 是否还残留 `outputs/`、`wandb/`、`__pycache__/`
- 是否还存在编辑器目录或其他本地痕迹

详细说明见：

- [`docs/release_checklist.md`](docs/release_checklist.md)

## 论文与实验记录

- 论文草稿：[`docs/paper_draft.md`](docs/paper_draft.md)
- 研究日志：[`docs/experiment_research_log.md`](docs/experiment_research_log.md)

## 说明

- 这个独立工程目录尽量保持可复现，但 `docs/` 中的历史记录和草稿仍然保留了部分原始工作区引用，便于回溯实验上下文。
- 主训练与图表脚本已经改成优先使用相对路径和环境变量，不再强依赖当前机器的绝对目录。
