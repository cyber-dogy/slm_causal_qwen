# Data1 Protocol Evaluation Report / Data1 协议评估报告

Generated / 生成时间: 2026-03-23 09:38:10

Scope / 协议范围: `official_cv`

Protocol Root / 协议目录: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol`

Model / 基础模型: `/home/gjw/code/model_cache/qwen/Qwen2-VL-2B-Instruct`

Prompt Alignment / Prompt 对齐方式: `train_validation`

2Class Run Root / 二分类运行目录: `/home/gjw/code/Qwen-SLM/runs/data1_official_cv_2class_baseline`

3Class Baseline Run Root / 三分类 Baseline 运行目录: `/home/gjw/code/Qwen-SLM/runs/data1_official_cv_3class_baseline`

3Class Causal Run Root / 三分类 Causal 运行目录: `/home/gjw/code/Qwen-SLM/runs/data1_official_cv_3class_causal`

## Metric Guide / 指标说明

- `accuracy`: overall accuracy / 总体准确率。
- `balanced_accuracy`: mean recall across classes / 各类别召回率平均，更适合不均衡分类。
- `macro_f1`: unweighted mean F1 across classes / 各类别 F1 的简单平均，推荐作为主指标。
- `weighted_f1`: support-weighted F1 / 按类别样本数加权的 F1。
- `unknown_rate`: unparsable prediction rate / 无法解析为合法标签的预测比例。
- `folded_2class`: fold `HEW/LEL` into `abnormal` / 将 `HEW/LEL` 折叠为 `abnormal` 的辅助异常识别指标。

- `prompt_alignment=train_validation`: use the same prompt-prefix truncation as train-time validation / 与训练验证使用同一条 prompt 前缀截断逻辑。

## Per-Fold Coverage / 各折覆盖结果

| Experiment | Scope | Fold | Stage | Mode | Split | Samples | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Unknown Rate |
|------------|-------|------|-------|------|-------|---------|----------|--------------|----------|-------------|--------------|
| 2class_baseline | official_cv | fold_01 | after | 2class | test | 95 | 0.6316 | 0.5000 | 0.3871 | 0.4890 | 0.0000 |
| 2class_baseline | official_cv | fold_01 | after | 2class | val | 39 | 0.4359 | 0.5000 | 0.3036 | 0.2647 | 0.0000 |
| 2class_baseline | official_cv | fold_01 | before | 2class | test | 95 | 0.6211 | 0.4917 | 0.3831 | 0.4839 | 0.0000 |
| 2class_baseline | official_cv | fold_01 | before | 2class | val | 39 | 0.4359 | 0.5000 | 0.3036 | 0.2647 | 0.0000 |
| 2class_baseline | official_cv | fold_02 | after | 2class | test | 98 | 0.6429 | 0.5000 | 0.3913 | 0.5031 | 0.0000 |
| 2class_baseline | official_cv | fold_02 | after | 2class | val | 46 | 0.5435 | 0.5000 | 0.3521 | 0.3827 | 0.0000 |
| 2class_baseline | official_cv | fold_02 | before | 2class | test | 98 | 0.6429 | 0.5000 | 0.3913 | 0.5031 | 0.0000 |
| 2class_baseline | official_cv | fold_02 | before | 2class | val | 46 | 0.5217 | 0.4800 | 0.3429 | 0.3727 | 0.0000 |
| 2class_baseline | official_cv | fold_03 | after | 2class | test | 96 | 0.6458 | 0.5000 | 0.3924 | 0.5069 | 0.0000 |
| 2class_baseline | official_cv | fold_03 | after | 2class | val | 60 | 0.5833 | 0.5000 | 0.3684 | 0.4298 | 0.0000 |
| 2class_baseline | official_cv | fold_03 | before | 2class | test | 96 | 0.6458 | 0.5000 | 0.3924 | 0.5069 | 0.0000 |
| 2class_baseline | official_cv | fold_03 | before | 2class | val | 60 | 0.5833 | 0.5000 | 0.3684 | 0.4298 | 0.0000 |
| 3class_baseline | official_cv | fold_01 | after | 3class | test | 95 | 0.5263 | 0.5524 | 0.4366 | 0.4219 | 0.0000 |
| 3class_baseline | official_cv | fold_01 | after | 3class | val | 39 | 0.3846 | 0.4221 | 0.3095 | 0.3315 | 0.0000 |
| 3class_baseline | official_cv | fold_01 | before | 3class | test | 95 | 0.3684 | 0.3333 | 0.1795 | 0.1984 | 0.0000 |
| 3class_baseline | official_cv | fold_01 | before | 3class | val | 39 | 0.5641 | 0.3333 | 0.2404 | 0.4069 | 0.0000 |
| 3class_baseline | official_cv | fold_02 | after | 3class | test | 98 | 0.5204 | 0.5128 | 0.4292 | 0.4351 | 0.0000 |
| 3class_baseline | official_cv | fold_02 | after | 3class | val | 46 | 0.6087 | 0.5397 | 0.4601 | 0.5272 | 0.0000 |
| 3class_baseline | official_cv | fold_02 | before | 3class | test | 98 | 0.3571 | 0.3333 | 0.1754 | 0.1880 | 0.0000 |
| 3class_baseline | official_cv | fold_02 | before | 3class | val | 46 | 0.4565 | 0.3333 | 0.2090 | 0.2862 | 0.0000 |
| 3class_baseline | official_cv | fold_03 | after | 3class | test | 96 | 0.3958 | 0.4412 | 0.3085 | 0.2850 | 0.0000 |
| 3class_baseline | official_cv | fold_03 | after | 3class | val | 60 | 0.4500 | 0.5200 | 0.3685 | 0.3587 | 0.0000 |
| 3class_baseline | official_cv | fold_03 | before | 3class | test | 96 | 0.3542 | 0.3333 | 0.1744 | 0.1853 | 0.0000 |
| 3class_baseline | official_cv | fold_03 | before | 3class | val | 60 | 0.4167 | 0.3333 | 0.1961 | 0.2451 | 0.0000 |
| 3class_causal | official_cv | fold_01 | after | 3class | test | 95 | 0.5053 | 0.5333 | 0.4170 | 0.4034 | 0.0000 |
| 3class_causal | official_cv | fold_01 | after | 3class | val | 39 | 0.3333 | 0.4281 | 0.3296 | 0.3048 | 0.0000 |
| 3class_causal | official_cv | fold_01 | before | 3class | test | 95 | 0.3684 | 0.3333 | 0.1795 | 0.1984 | 0.0000 |
| 3class_causal | official_cv | fold_01 | before | 3class | val | 39 | 0.5641 | 0.3333 | 0.2404 | 0.4069 | 0.0000 |
| 3class_causal | official_cv | fold_02 | after | 3class | test | 98 | 0.4388 | 0.4363 | 0.3333 | 0.3386 | 0.0000 |
| 3class_causal | official_cv | fold_02 | after | 3class | val | 46 | 0.4565 | 0.4286 | 0.3250 | 0.3668 | 0.0000 |
| 3class_causal | official_cv | fold_02 | before | 3class | test | 98 | 0.3571 | 0.3333 | 0.1754 | 0.1880 | 0.0000 |
| 3class_causal | official_cv | fold_02 | before | 3class | val | 46 | 0.4565 | 0.3333 | 0.2090 | 0.2862 | 0.0000 |
| 3class_causal | official_cv | fold_03 | after | 3class | test | 96 | 0.3542 | 0.4020 | 0.2616 | 0.2389 | 0.0000 |
| 3class_causal | official_cv | fold_03 | after | 3class | val | 60 | 0.4333 | 0.5067 | 0.3561 | 0.3470 | 0.0000 |
| 3class_causal | official_cv | fold_03 | before | 3class | test | 96 | 0.3542 | 0.3333 | 0.1744 | 0.1853 | 0.0000 |
| 3class_causal | official_cv | fold_03 | before | 3class | val | 60 | 0.4167 | 0.3333 | 0.1961 | 0.2451 | 0.0000 |

## Aggregate Mean ± Std / 汇总均值 ± 标准差

| Experiment | Stage | Mode | Split | Folds | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Unknown Rate |
|------------|-------|------|-------|-------|----------|--------------|----------|-------------|--------------|
| 2class_baseline | after | 2class | test | 3 | 0.6401 ± 0.0061 | 0.5000 ± 0.0000 | 0.3903 ± 0.0023 | 0.4996 ± 0.0077 | 0.0000 ± 0.0000 |
| 2class_baseline | after | 2class | val | 3 | 0.5209 ± 0.0623 | 0.5000 ± 0.0000 | 0.3414 ± 0.0275 | 0.3591 ± 0.0695 | 0.0000 ± 0.0000 |
| 2class_baseline | before | 2class | test | 3 | 0.6366 ± 0.0110 | 0.4972 ± 0.0039 | 0.3889 ± 0.0041 | 0.4980 ± 0.0100 | 0.0000 ± 0.0000 |
| 2class_baseline | before | 2class | val | 3 | 0.5137 ± 0.0605 | 0.4933 ± 0.0094 | 0.3383 ± 0.0267 | 0.3557 ± 0.0685 | 0.0000 ± 0.0000 |
| 3class_baseline | after | 3class | test | 3 | 0.4809 ± 0.0602 | 0.5021 ± 0.0460 | 0.3914 ± 0.0587 | 0.3806 ± 0.0679 | 0.0000 ± 0.0000 |
| 3class_baseline | after | 3class | val | 3 | 0.4811 ± 0.0941 | 0.4939 ± 0.0514 | 0.3794 ± 0.0620 | 0.4058 ± 0.0866 | 0.0000 ± 0.0000 |
| 3class_baseline | before | 3class | test | 3 | 0.3599 ± 0.0061 | 0.3333 ± 0.0000 | 0.1764 ± 0.0022 | 0.1905 ± 0.0057 | 0.0000 ± 0.0000 |
| 3class_baseline | before | 3class | val | 3 | 0.4791 ± 0.0623 | 0.3333 ± 0.0000 | 0.2152 ± 0.0186 | 0.3127 ± 0.0687 | 0.0000 ± 0.0000 |
| 3class_causal | after | 3class | test | 3 | 0.4327 ± 0.0618 | 0.4572 ± 0.0556 | 0.3373 ± 0.0635 | 0.3270 ± 0.0676 | 0.0000 ± 0.0000 |
| 3class_causal | after | 3class | val | 3 | 0.4077 ± 0.0535 | 0.4545 ± 0.0369 | 0.3369 ± 0.0137 | 0.3396 ± 0.0259 | 0.0000 ± 0.0000 |
| 3class_causal | before | 3class | test | 3 | 0.3599 ± 0.0061 | 0.3333 ± 0.0000 | 0.1764 ± 0.0022 | 0.1905 ± 0.0057 | 0.0000 ± 0.0000 |
| 3class_causal | before | 3class | val | 3 | 0.4791 ± 0.0623 | 0.3333 ± 0.0000 | 0.2152 ± 0.0186 | 0.3127 ± 0.0687 | 0.0000 ± 0.0000 |

## Before vs After / 微调前后对比

### 2Class Baseline / 二分类 Baseline / val

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| accuracy | 0.5137 | 0.5209 | 0.0072 |
| balanced_accuracy | 0.4933 | 0.5000 | 0.0067 |
| macro_f1 | 0.3383 | 0.3414 | 0.0031 |
| weighted_f1 | 0.3557 | 0.3591 | 0.0034 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

### 2Class Baseline / 二分类 Baseline / test

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| accuracy | 0.6366 | 0.6401 | 0.0035 |
| balanced_accuracy | 0.4972 | 0.5000 | 0.0028 |
| macro_f1 | 0.3889 | 0.3903 | 0.0013 |
| weighted_f1 | 0.4980 | 0.4996 | 0.0017 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

### 3Class Baseline / 三分类 Baseline / val

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| accuracy | 0.4791 | 0.4811 | 0.0020 |
| balanced_accuracy | 0.3333 | 0.4939 | 0.1606 |
| macro_f1 | 0.2152 | 0.3794 | 0.1642 |
| weighted_f1 | 0.3127 | 0.4058 | 0.0931 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| folded::accuracy | 0.4791 | 0.5589 | 0.0798 |
| folded::balanced_accuracy | 0.5000 | 0.5521 | 0.0521 |
| folded::macro_f1 | 0.3227 | 0.5522 | 0.2295 |
| folded::weighted_f1 | 0.3127 | 0.5587 | 0.2460 |
| folded::unknown_rate | 0.0000 | 0.0000 | 0.0000 |

### 3Class Baseline / 三分类 Baseline / test

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| accuracy | 0.3599 | 0.4809 | 0.1209 |
| balanced_accuracy | 0.3333 | 0.5021 | 0.1688 |
| macro_f1 | 0.1764 | 0.3914 | 0.2150 |
| weighted_f1 | 0.1905 | 0.3806 | 0.1901 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| folded::accuracy | 0.3599 | 0.5956 | 0.2356 |
| folded::balanced_accuracy | 0.5000 | 0.5801 | 0.0801 |
| folded::macro_f1 | 0.2646 | 0.5715 | 0.3068 |
| folded::weighted_f1 | 0.1905 | 0.5986 | 0.4081 |
| folded::unknown_rate | 0.0000 | 0.0000 | 0.0000 |

### 3Class Causal / 三分类 Causal / val

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| accuracy | 0.4791 | 0.4077 | -0.0714 |
| balanced_accuracy | 0.3333 | 0.4545 | 0.1211 |
| macro_f1 | 0.2152 | 0.3369 | 0.1218 |
| weighted_f1 | 0.3127 | 0.3396 | 0.0269 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| folded::accuracy | 0.4791 | 0.5273 | 0.0482 |
| folded::balanced_accuracy | 0.5000 | 0.5173 | 0.0173 |
| folded::macro_f1 | 0.3227 | 0.5041 | 0.1814 |
| folded::weighted_f1 | 0.3127 | 0.5109 | 0.1981 |
| folded::unknown_rate | 0.0000 | 0.0000 | 0.0000 |

### 3Class Causal / 三分类 Causal / test

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| accuracy | 0.3599 | 0.4327 | 0.0728 |
| balanced_accuracy | 0.3333 | 0.4572 | 0.1239 |
| macro_f1 | 0.1764 | 0.3373 | 0.1609 |
| weighted_f1 | 0.1905 | 0.3270 | 0.1364 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric | Before Mean | After Mean | Delta |
|--------|-------------|------------|-------|
| folded::accuracy | 0.3599 | 0.6058 | 0.2459 |
| folded::balanced_accuracy | 0.5000 | 0.5583 | 0.0583 |
| folded::macro_f1 | 0.2646 | 0.5532 | 0.2885 |
| folded::weighted_f1 | 0.1905 | 0.5957 | 0.4052 |
| folded::unknown_rate | 0.0000 | 0.0000 | 0.0000 |

## 2Class Baseline vs 3Class Variants / 二分类 Baseline 与三分类变体对比

### before / val / 3Class Baseline / 三分类 Baseline

| Metric | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
|--------|-------------|------------------|--------------------|----------------------|
| accuracy | 0.5137 | 0.4791 | 0.4791 | 0.0346 |
| balanced_accuracy | 0.4933 | 0.3333 | 0.5000 | -0.0067 |
| macro_f1 | 0.3383 | 0.2152 | 0.3227 | 0.0155 |
| weighted_f1 | 0.3557 | 0.3127 | 0.3127 | 0.0430 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### before / val / 3Class Causal / 三分类 Causal

| Metric | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
|--------|-------------|------------------|--------------------|----------------------|
| accuracy | 0.5137 | 0.4791 | 0.4791 | 0.0346 |
| balanced_accuracy | 0.4933 | 0.3333 | 0.5000 | -0.0067 |
| macro_f1 | 0.3383 | 0.2152 | 0.3227 | 0.0155 |
| weighted_f1 | 0.3557 | 0.3127 | 0.3127 | 0.0430 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### before / test / 3Class Baseline / 三分类 Baseline

| Metric | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
|--------|-------------|------------------|--------------------|----------------------|
| accuracy | 0.6366 | 0.3599 | 0.3599 | 0.2767 |
| balanced_accuracy | 0.4972 | 0.3333 | 0.5000 | -0.0028 |
| macro_f1 | 0.3889 | 0.1764 | 0.2646 | 0.1243 |
| weighted_f1 | 0.4980 | 0.1905 | 0.1905 | 0.3074 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### before / test / 3Class Causal / 三分类 Causal

| Metric | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
|--------|-------------|------------------|--------------------|----------------------|
| accuracy | 0.6366 | 0.3599 | 0.3599 | 0.2767 |
| balanced_accuracy | 0.4972 | 0.3333 | 0.5000 | -0.0028 |
| macro_f1 | 0.3889 | 0.1764 | 0.2646 | 0.1243 |
| weighted_f1 | 0.4980 | 0.1905 | 0.1905 | 0.3074 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### after / val / 3Class Baseline / 三分类 Baseline

| Metric | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
|--------|-------------|------------------|--------------------|----------------------|
| accuracy | 0.5209 | 0.4811 | 0.5589 | -0.0380 |
| balanced_accuracy | 0.5000 | 0.4939 | 0.5521 | -0.0521 |
| macro_f1 | 0.3414 | 0.3794 | 0.5522 | -0.2108 |
| weighted_f1 | 0.3591 | 0.4058 | 0.5587 | -0.1997 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### after / val / 3Class Causal / 三分类 Causal

| Metric | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
|--------|-------------|------------------|--------------------|----------------------|
| accuracy | 0.5209 | 0.4077 | 0.5273 | -0.0064 |
| balanced_accuracy | 0.5000 | 0.4545 | 0.5173 | -0.0173 |
| macro_f1 | 0.3414 | 0.3369 | 0.5041 | -0.1628 |
| weighted_f1 | 0.3591 | 0.3396 | 0.5109 | -0.1518 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### after / test / 3Class Baseline / 三分类 Baseline

| Metric | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
|--------|-------------|------------------|--------------------|----------------------|
| accuracy | 0.6401 | 0.4809 | 0.5956 | 0.0445 |
| balanced_accuracy | 0.5000 | 0.5021 | 0.5801 | -0.0801 |
| macro_f1 | 0.3903 | 0.3914 | 0.5715 | -0.1812 |
| weighted_f1 | 0.4996 | 0.3806 | 0.5986 | -0.0990 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### after / test / 3Class Causal / 三分类 Causal

| Metric | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
|--------|-------------|------------------|--------------------|----------------------|
| accuracy | 0.6401 | 0.4327 | 0.6058 | 0.0343 |
| balanced_accuracy | 0.5000 | 0.4572 | 0.5583 | -0.0583 |
| macro_f1 | 0.3903 | 0.3373 | 0.5532 | -0.1629 |
| weighted_f1 | 0.4996 | 0.3270 | 0.5957 | -0.0961 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 3Class Baseline vs 3Class Causal / 三分类 Baseline 与 Causal 对比

### before / val

| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
|--------|---------------|-------------|------------------------|
| accuracy | 0.4791 | 0.4791 | 0.0000 |
| balanced_accuracy | 0.3333 | 0.3333 | 0.0000 |
| macro_f1 | 0.2152 | 0.2152 | 0.0000 |
| weighted_f1 | 0.3127 | 0.3127 | 0.0000 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
|--------|---------------|-------------|------------------------|
| folded::accuracy | 0.4791 | 0.4791 | 0.0000 |
| folded::balanced_accuracy | 0.5000 | 0.5000 | 0.0000 |
| folded::macro_f1 | 0.3227 | 0.3227 | 0.0000 |
| folded::weighted_f1 | 0.3127 | 0.3127 | 0.0000 |
| folded::unknown_rate | 0.0000 | 0.0000 | 0.0000 |

### before / test

| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
|--------|---------------|-------------|------------------------|
| accuracy | 0.3599 | 0.3599 | 0.0000 |
| balanced_accuracy | 0.3333 | 0.3333 | 0.0000 |
| macro_f1 | 0.1764 | 0.1764 | 0.0000 |
| weighted_f1 | 0.1905 | 0.1905 | 0.0000 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
|--------|---------------|-------------|------------------------|
| folded::accuracy | 0.3599 | 0.3599 | 0.0000 |
| folded::balanced_accuracy | 0.5000 | 0.5000 | 0.0000 |
| folded::macro_f1 | 0.2646 | 0.2646 | 0.0000 |
| folded::weighted_f1 | 0.1905 | 0.1905 | 0.0000 |
| folded::unknown_rate | 0.0000 | 0.0000 | 0.0000 |

### after / val

| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
|--------|---------------|-------------|------------------------|
| accuracy | 0.4811 | 0.4077 | -0.0734 |
| balanced_accuracy | 0.4939 | 0.4545 | -0.0395 |
| macro_f1 | 0.3794 | 0.3369 | -0.0425 |
| weighted_f1 | 0.4058 | 0.3396 | -0.0662 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
|--------|---------------|-------------|------------------------|
| folded::accuracy | 0.5589 | 0.5273 | -0.0316 |
| folded::balanced_accuracy | 0.5521 | 0.5173 | -0.0348 |
| folded::macro_f1 | 0.5522 | 0.5041 | -0.0481 |
| folded::weighted_f1 | 0.5587 | 0.5109 | -0.0479 |
| folded::unknown_rate | 0.0000 | 0.0000 | 0.0000 |

### after / test

| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
|--------|---------------|-------------|------------------------|
| accuracy | 0.4809 | 0.4327 | -0.0481 |
| balanced_accuracy | 0.5021 | 0.4572 | -0.0449 |
| macro_f1 | 0.3914 | 0.3373 | -0.0541 |
| weighted_f1 | 0.3806 | 0.3270 | -0.0537 |
| unknown_rate | 0.0000 | 0.0000 | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
|--------|---------------|-------------|------------------------|
| folded::accuracy | 0.5956 | 0.6058 | 0.0103 |
| folded::balanced_accuracy | 0.5801 | 0.5583 | -0.0218 |
| folded::macro_f1 | 0.5715 | 0.5532 | -0.0183 |
| folded::weighted_f1 | 0.5986 | 0.5957 | -0.0029 |
| folded::unknown_rate | 0.0000 | 0.0000 | 0.0000 |

