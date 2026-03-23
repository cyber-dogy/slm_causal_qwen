# 发布前检查清单

这份清单用于把当前工程整理成适合上传 GitHub 的论文复现项目。

## 1. 必须确认不上传的内容

下面这些内容应保持本地存在，但不要提交到 GitHub：

- `configs/demo.local.yaml`
- `configs/local_paths.local.env`
- `outputs/`
- `wandb/`
- 数据集原图目录
- 训练产物、权重、checkpoint、adapter
- `paper_visualization/cache/`
- `paper_visualization/exports/`

## 2. 本地痕迹检查

需要避免残留：

- `.git/`（嵌套仓库痕迹）
- `.vscode/`
- `.idea/`
- `__pycache__/`
- `*.pyc`
- `*.pyo`
- `.DS_Store`

## 3. 推荐自查命令

```bash
bash scripts/check_release_tree.sh
```

如果脚本输出仍有本地文件，请在上传前手动确认。

## 4. README 需要包含什么

为了让工程可复现，主 README 至少要包含：

- 环境准备
- 自训练路径
- 已训练权重的接入路径
- demo 启动方式
- 本地路径模板说明

## 5. 权重发布建议

如果后续通过 Quark 发布权重，建议同步提供：

- 下载链接
- 提取码
- 版本说明
- 推荐目录结构
- `specialized.run_root` 的配置示例

这些信息已经预留在：

- [`model_weights.md`](model_weights.md)
