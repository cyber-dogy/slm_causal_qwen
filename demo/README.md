# Demo

这个目录提供一个对比式 Gradio demo：

- 左侧：专用 `QwenVisFusion + causal consistency` 模型
- 右侧：通用 `Qwen2-VL` 单图 VQA 模型

右侧通用分支是对你此前 `industrial_vqa_demo` 思路的自包含复刻，
这样当前论文工程不需要依赖外部仓库也能直接展示对比效果。

## 1. 准备本地配置

复制示例配置：

```bash
cp configs/demo.example.yaml configs/demo.local.yaml
```

然后把这些路径改成你本地可用的：

- `specialized.run_root` 或 `specialized.checkpoint_path`
- `general.model_name_or_path`

`specialized.run_root` 有两种来源：

1. 你自己训练得到的本地实验目录
2. 后续从 Quark 网盘下载并解压得到的发布目录

如果你已经在当前工作区跑出了最佳模型，可以把 `specialized.run_root`
直接指向本地最佳实验目录，例如 `official_cv_3class_rgb_dual_causal`。

权重说明见：

- [`../docs/model_weights.md`](../docs/model_weights.md)

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 启动

```bash
python demo/app.py --config configs/demo.local.yaml
```

## 4. 使用说明

- `RGB View 1` 和 `RGB View 2` 是专用模型的必需输入
- `IR After` 在当前最佳专用模型里不会参与分类，但会拼入右侧通用 VQA 的 montage
- 专用模型会输出结构化概率和缺陷类别
- 通用模型会给出自然语言回答，方便和你之前的通用 demo 直观对比
- 如果你后续下载的是 Quark 发布权重，只需要把 `specialized.run_root` 指向解压目录，不需要改 demo 代码
