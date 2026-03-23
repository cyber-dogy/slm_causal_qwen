# 模型权重与自训练说明

这个工程默认不包含任何模型参数、checkpoint 或 adapter。你可以通过下面两种方式使用专用模型。

## 方式 A：自行训练

这是最推荐的复现路径，也是论文工程默认假设的使用方式。

### 1. 准备本地路径

复制并填写：

```bash
cp configs/local_paths.example.env configs/local_paths.local.env
source configs/local_paths.local.env
```

至少需要：

- `SLM_DATA1_PROTOCOL_ROOT`
- `SLM_DATA2_TRANSFER_ROOT`
- `QWEN2_VL_MODEL_PATH`
- `PROJECT_OUTPUT_ROOT`

### 2. 训练论文主方法

```bash
python qwen_vis_fusion/train_cv.py \
  --config qwen_vis_fusion/configs/official_cv/3class_rgb_dual_causal.json
```

训练完成后，通常会得到类似目录：

```text
outputs/qwen_vis_fusion/official_cv_3class_rgb_dual_causal/
  run_summary.json
  fold_01/
    best.pt
    config.json
  fold_02/
    best.pt
    config.json
  fold_03/
    best.pt
    config.json
```

### 3. 接到 demo

把 demo 配置里的：

- `specialized.run_root`

指向上面的运行目录，例如：

```yaml
specialized:
  run_root: "/your/path/to/outputs/qwen_vis_fusion/official_cv_3class_rgb_dual_causal"
  fold: "best"
```

然后启动：

```bash
python demo/app.py --config configs/demo.local.yaml
```

## 方式 B：使用后续发布的 Quark 权重

后续作者会把专用模型的 demo 运行目录整理后上传到 Quark 网盘。这里预留使用说明。

### 预留信息

- Quark 分享链接：`TODO`
- 提取码：`TODO`
- 发布版本说明：`TODO`

### 推荐发布内容

建议发布的是可直接给 demo 使用的运行目录，而不是零散单个权重文件。推荐结构：

```text
official_cv_3class_rgb_dual_causal/
  run_summary.json
  fold_01/
    best.pt
    config.json
  fold_02/
    best.pt
    config.json
  fold_03/
    best.pt
    config.json
```

### 下载后如何使用

1. 从 Quark 下载并解压
2. 修改 `configs/demo.local.yaml`
3. 把：

- `specialized.run_root`

改成解压后的目录

例如：

```yaml
specialized:
  run_root: "/your/path/to/official_cv_3class_rgb_dual_causal"
  fold: "best"
```

然后直接启动 demo：

```bash
python demo/app.py --config configs/demo.local.yaml
```

## 经典 baseline 权重

经典 baseline 不是 demo 的必要依赖。如果后续也要发布，可以参考主方法同样的目录组织方式，单独整理到：

- `classic_mm_baselines/...`

但当前工程并不强依赖这些权重。

## 建议

对外发布时，建议 README 里同时保留：

1. 自训练路径
2. Quark 下载路径

这样即使网盘链接失效，工程仍然具备可复现性。
