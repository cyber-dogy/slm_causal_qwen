# Data1 Protocol Evaluation Report / Data1 协议评估报告

Generated / 生成时间: 2026-03-23 06:42:19

Scope / 协议范围: `official_cv`

Protocol Root / 协议目录: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol`

Model / 基础模型: `/home/gjw/code/model_cache/qwen/Qwen2-VL-2B-Instruct`

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

## Per-Fold Coverage / 各折覆盖结果

| Experiment      | Scope       | Fold    | Stage  | Mode   | Split | Samples | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Unknown Rate |
| --------------- | ----------- | ------- | ------ | ------ | ----- | ------- | -------- | ------------ | -------- | ----------- | ------------ |
| 2class_baseline | official_cv | fold_01 | after  | 2class | test  | 95      | 0.6316   | 0.5000       | 0.3871   | 0.4890      | 0.0000       |
| 2class_baseline | official_cv | fold_01 | after  | 2class | val   | 39      | 0.4359   | 0.5000       | 0.3036   | 0.2647      | 0.0000       |
| 2class_baseline | official_cv | fold_01 | before | 2class | test  | 95      | 0.6316   | 0.5000       | 0.3871   | 0.4890      | 0.0000       |
| 2class_baseline | official_cv | fold_01 | before | 2class | val   | 39      | 0.4359   | 0.5000       | 0.3036   | 0.2647      | 0.0000       |
| 2class_baseline | official_cv | fold_02 | after  | 2class | test  | 98      | 0.6429   | 0.5000       | 0.3913   | 0.5031      | 0.0000       |
| 2class_baseline | official_cv | fold_02 | after  | 2class | val   | 46      | 0.5435   | 0.5000       | 0.3521   | 0.3827      | 0.0000       |
| 2class_baseline | official_cv | fold_02 | before | 2class | test  | 98      | 0.6429   | 0.5000       | 0.3913   | 0.5031      | 0.0000       |
| 2class_baseline | official_cv | fold_02 | before | 2class | val   | 46      | 0.5435   | 0.5000       | 0.3521   | 0.3827      | 0.0000       |
| 2class_baseline | official_cv | fold_03 | after  | 2class | test  | 96      | 0.6458   | 0.5000       | 0.3924   | 0.5069      | 0.0000       |
| 2class_baseline | official_cv | fold_03 | after  | 2class | val   | 60      | 0.5833   | 0.5000       | 0.3684   | 0.4298      | 0.0000       |
| 2class_baseline | official_cv | fold_03 | before | 2class | test  | 96      | 0.6458   | 0.5000       | 0.3924   | 0.5069      | 0.0000       |
| 2class_baseline | official_cv | fold_03 | before | 2class | val   | 60      | 0.5833   | 0.5000       | 0.3684   | 0.4298      | 0.0000       |
| 3class_baseline | official_cv | fold_01 | after  | 3class | test  | 95      | 0.3684   | 0.3524       | 0.2106   | 0.2218      | 0.0000       |
| 3class_baseline | official_cv | fold_01 | after  | 3class | val   | 39      | 0.2821   | 0.3485       | 0.1679   | 0.1559      | 0.0000       |
| 3class_baseline | official_cv | fold_01 | before | 3class | test  | 95      | 0.3684   | 0.3524       | 0.2106   | 0.2218      | 0.0000       |
| 3class_baseline | official_cv | fold_01 | before | 3class | val   | 39      | 0.2821   | 0.3485       | 0.1679   | 0.1559      | 0.0000       |
| 3class_baseline | official_cv | fold_02 | after  | 3class | test  | 98      | 0.3265   | 0.3429       | 0.1800   | 0.1731      | 0.0000       |
| 3class_baseline | official_cv | fold_02 | after  | 3class | val   | 46      | 0.2174   | 0.3333       | 0.1190   | 0.0776      | 0.0000       |
| 3class_baseline | official_cv | fold_02 | before | 3class | test  | 98      | 0.3265   | 0.3429       | 0.1800   | 0.1731      | 0.0000       |
| 3class_baseline | official_cv | fold_02 | before | 3class | val   | 46      | 0.2174   | 0.3333       | 0.1190   | 0.0776      | 0.0000       |
| 3class_baseline | official_cv | fold_03 | after  | 3class | test  | 96      | 0.3646   | 0.3333       | 0.1781   | 0.1948      | 0.0000       |
| 3class_baseline | official_cv | fold_03 | after  | 3class | val   | 60      | 0.4000   | 0.3600       | 0.2327   | 0.2634      | 0.0000       |
| 3class_baseline | official_cv | fold_03 | before | 3class | test  | 96      | 0.3646   | 0.3333       | 0.1781   | 0.1948      | 0.0000       |
| 3class_baseline | official_cv | fold_03 | before | 3class | val   | 60      | 0.4000   | 0.3600       | 0.2327   | 0.2634      | 0.0000       |
| 3class_causal   | official_cv | fold_01 | after  | 3class | test  | 95      | 0.3684   | 0.3524       | 0.2106   | 0.2218      | 0.0000       |
| 3class_causal   | official_cv | fold_01 | after  | 3class | val   | 39      | 0.2821   | 0.3485       | 0.1679   | 0.1559      | 0.0000       |
| 3class_causal   | official_cv | fold_01 | before | 3class | test  | 95      | 0.3684   | 0.3524       | 0.2106   | 0.2218      | 0.0000       |
| 3class_causal   | official_cv | fold_01 | before | 3class | val   | 39      | 0.2821   | 0.3485       | 0.1679   | 0.1559      | 0.0000       |
| 3class_causal   | official_cv | fold_02 | after  | 3class | test  | 98      | 0.3265   | 0.3429       | 0.1800   | 0.1731      | 0.0000       |
| 3class_causal   | official_cv | fold_02 | after  | 3class | val   | 46      | 0.2174   | 0.3333       | 0.1190   | 0.0776      | 0.0000       |
| 3class_causal   | official_cv | fold_02 | before | 3class | test  | 98      | 0.3265   | 0.3429       | 0.1800   | 0.1731      | 0.0000       |
| 3class_causal   | official_cv | fold_02 | before | 3class | val   | 46      | 0.2174   | 0.3333       | 0.1190   | 0.0776      | 0.0000       |
| 3class_causal   | official_cv | fold_03 | after  | 3class | test  | 96      | 0.3646   | 0.3333       | 0.1781   | 0.1948      | 0.0000       |
| 3class_causal   | official_cv | fold_03 | after  | 3class | val   | 60      | 0.3833   | 0.3467       | 0.2067   | 0.2312      | 0.0000       |
| 3class_causal   | official_cv | fold_03 | before | 3class | test  | 96      | 0.3646   | 0.3333       | 0.1781   | 0.1948      | 0.0000       |
| 3class_causal   | official_cv | fold_03 | before | 3class | val   | 60      | 0.4000   | 0.3600       | 0.2327   | 0.2634      | 0.0000       |

## Aggregate Mean ± Std / 汇总均值 ± 标准差

| Experiment      | Stage  | Mode   | Split | Folds | Accuracy         | Balanced Acc     | Macro F1         | Weighted F1      | Unknown Rate     |
| --------------- | ------ | ------ | ----- | ----- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 2class_baseline | after  | 2class | test  | 3     | 0.6401 ± 0.0061 | 0.5000 ± 0.0000 | 0.3903 ± 0.0023 | 0.4996 ± 0.0077 | 0.0000 ± 0.0000 |
| 2class_baseline | after  | 2class | val   | 3     | 0.5209 ± 0.0623 | 0.5000 ± 0.0000 | 0.3414 ± 0.0275 | 0.3591 ± 0.0695 | 0.0000 ± 0.0000 |
| 2class_baseline | before | 2class | test  | 3     | 0.6401 ± 0.0061 | 0.5000 ± 0.0000 | 0.3903 ± 0.0023 | 0.4996 ± 0.0077 | 0.0000 ± 0.0000 |
| 2class_baseline | before | 2class | val   | 3     | 0.5209 ± 0.0623 | 0.5000 ± 0.0000 | 0.3414 ± 0.0275 | 0.3591 ± 0.0695 | 0.0000 ± 0.0000 |
| 3class_baseline | after  | 3class | test  | 3     | 0.3532 ± 0.0189 | 0.3429 ± 0.0078 | 0.1896 ± 0.0149 | 0.1966 ± 0.0199 | 0.0000 ± 0.0000 |
| 3class_baseline | after  | 3class | val   | 3     | 0.2998 ± 0.0756 | 0.3473 ± 0.0109 | 0.1732 ± 0.0466 | 0.1656 ± 0.0761 | 0.0000 ± 0.0000 |
| 3class_baseline | before | 3class | test  | 3     | 0.3532 ± 0.0189 | 0.3429 ± 0.0078 | 0.1896 ± 0.0149 | 0.1966 ± 0.0199 | 0.0000 ± 0.0000 |
| 3class_baseline | before | 3class | val   | 3     | 0.2998 ± 0.0756 | 0.3473 ± 0.0109 | 0.1732 ± 0.0466 | 0.1656 ± 0.0761 | 0.0000 ± 0.0000 |
| 3class_causal   | after  | 3class | test  | 3     | 0.3532 ± 0.0189 | 0.3429 ± 0.0078 | 0.1896 ± 0.0149 | 0.1966 ± 0.0199 | 0.0000 ± 0.0000 |
| 3class_causal   | after  | 3class | val   | 3     | 0.2943 ± 0.0683 | 0.3428 ± 0.0068 | 0.1645 ± 0.0359 | 0.1549 ± 0.0627 | 0.0000 ± 0.0000 |
| 3class_causal   | before | 3class | test  | 3     | 0.3532 ± 0.0189 | 0.3429 ± 0.0078 | 0.1896 ± 0.0149 | 0.1966 ± 0.0199 | 0.0000 ± 0.0000 |
| 3class_causal   | before | 3class | val   | 3     | 0.2998 ± 0.0756 | 0.3473 ± 0.0109 | 0.1732 ± 0.0466 | 0.1656 ± 0.0761 | 0.0000 ± 0.0000 |

## Before vs After / 微调前后对比

### 2Class Baseline / 二分类 Baseline / val

| Metric            | Before Mean | After Mean | Delta  |
| ----------------- | ----------- | ---------- | ------ |
| accuracy          | 0.5209      | 0.5209     | 0.0000 |
| balanced_accuracy | 0.5000      | 0.5000     | 0.0000 |
| macro_f1          | 0.3414      | 0.3414     | 0.0000 |
| weighted_f1       | 0.3591      | 0.3591     | 0.0000 |
| unknown_rate      | 0.0000      | 0.0000     | 0.0000 |

### 2Class Baseline / 二分类 Baseline / test

| Metric            | Before Mean | After Mean | Delta  |
| ----------------- | ----------- | ---------- | ------ |
| accuracy          | 0.6401      | 0.6401     | 0.0000 |
| balanced_accuracy | 0.5000      | 0.5000     | 0.0000 |
| macro_f1          | 0.3903      | 0.3903     | 0.0000 |
| weighted_f1       | 0.4996      | 0.4996     | 0.0000 |
| unknown_rate      | 0.0000      | 0.0000     | 0.0000 |

### 3Class Baseline / 三分类 Baseline / val

| Metric            | Before Mean | After Mean | Delta  |
| ----------------- | ----------- | ---------- | ------ |
| accuracy          | 0.2998      | 0.2998     | 0.0000 |
| balanced_accuracy | 0.3473      | 0.3473     | 0.0000 |
| macro_f1          | 0.1732      | 0.1732     | 0.0000 |
| weighted_f1       | 0.1656      | 0.1656     | 0.0000 |
| unknown_rate      | 0.0000      | 0.0000     | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric                    | Before Mean | After Mean | Delta  |
| ------------------------- | ----------- | ---------- | ------ |
| folded::accuracy          | 0.5406      | 0.5406     | 0.0000 |
| folded::balanced_accuracy | 0.5209      | 0.5209     | 0.0000 |
| folded::macro_f1          | 0.3850      | 0.3850     | 0.0000 |
| folded::weighted_f1       | 0.4007      | 0.4007     | 0.0000 |
| folded::unknown_rate      | 0.0000      | 0.0000     | 0.0000 |

### 3Class Baseline / 三分类 Baseline / test

| Metric            | Before Mean | After Mean | Delta  |
| ----------------- | ----------- | ---------- | ------ |
| accuracy          | 0.3532      | 0.3532     | 0.0000 |
| balanced_accuracy | 0.3429      | 0.3429     | 0.0000 |
| macro_f1          | 0.1896      | 0.1896     | 0.0000 |
| weighted_f1       | 0.1966      | 0.1966     | 0.0000 |
| unknown_rate      | 0.0000      | 0.0000     | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric                    | Before Mean | After Mean | Delta  |
| ------------------------- | ----------- | ---------- | ------ |
| folded::accuracy          | 0.6505      | 0.6505     | 0.0000 |
| folded::balanced_accuracy | 0.5143      | 0.5143     | 0.0000 |
| folded::macro_f1          | 0.4200      | 0.4200     | 0.0000 |
| folded::weighted_f1       | 0.5227      | 0.5227     | 0.0000 |
| folded::unknown_rate      | 0.0000      | 0.0000     | 0.0000 |

### 3Class Causal / 三分类 Causal / val

| Metric            | Before Mean | After Mean | Delta   |
| ----------------- | ----------- | ---------- | ------- |
| accuracy          | 0.2998      | 0.2943     | -0.0056 |
| balanced_accuracy | 0.3473      | 0.3428     | -0.0044 |
| macro_f1          | 0.1732      | 0.1645     | -0.0087 |
| weighted_f1       | 0.1656      | 0.1549     | -0.0107 |
| unknown_rate      | 0.0000      | 0.0000     | 0.0000  |

Folded 2Class / 三分类折叠成二分类

| Metric                    | Before Mean | After Mean | Delta   |
| ------------------------- | ----------- | ---------- | ------- |
| folded::accuracy          | 0.5406      | 0.5350     | -0.0056 |
| folded::balanced_accuracy | 0.5209      | 0.5142     | -0.0067 |
| folded::macro_f1          | 0.3850      | 0.3718     | -0.0132 |
| folded::weighted_f1       | 0.4007      | 0.3892     | -0.0114 |
| folded::unknown_rate      | 0.0000      | 0.0000     | 0.0000  |

### 3Class Causal / 三分类 Causal / test

| Metric            | Before Mean | After Mean | Delta  |
| ----------------- | ----------- | ---------- | ------ |
| accuracy          | 0.3532      | 0.3532     | 0.0000 |
| balanced_accuracy | 0.3429      | 0.3429     | 0.0000 |
| macro_f1          | 0.1896      | 0.1896     | 0.0000 |
| weighted_f1       | 0.1966      | 0.1966     | 0.0000 |
| unknown_rate      | 0.0000      | 0.0000     | 0.0000 |

Folded 2Class / 三分类折叠成二分类

| Metric                    | Before Mean | After Mean | Delta  |
| ------------------------- | ----------- | ---------- | ------ |
| folded::accuracy          | 0.6505      | 0.6505     | 0.0000 |
| folded::balanced_accuracy | 0.5143      | 0.5143     | 0.0000 |
| folded::macro_f1          | 0.4200      | 0.4200     | 0.0000 |
| folded::weighted_f1       | 0.5227      | 0.5227     | 0.0000 |
| folded::unknown_rate      | 0.0000      | 0.0000     | 0.0000 |

## 2Class Baseline vs 3Class Variants / 二分类 Baseline 与三分类变体对比

### before / val / 3Class Baseline / 三分类 Baseline

| Metric            | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
| ----------------- | -------------------- | ---------------- | ------------------ | -------------------- |
| accuracy          | 0.5209               | 0.2998           | 0.5406             | -0.0197              |
| balanced_accuracy | 0.5000               | 0.3473           | 0.5209             | -0.0209              |
| macro_f1          | 0.3414               | 0.1732           | 0.3850             | -0.0437              |
| weighted_f1       | 0.3591               | 0.1656           | 0.4007             | -0.0416              |
| unknown_rate      | 0.0000               | 0.0000           | 0.0000             | 0.0000               |

### before / val / 3Class Causal / 三分类 Causal

| Metric            | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
| ----------------- | -------------------- | ---------------- | ------------------ | -------------------- |
| accuracy          | 0.5209               | 0.2998           | 0.5406             | -0.0197              |
| balanced_accuracy | 0.5000               | 0.3473           | 0.5209             | -0.0209              |
| macro_f1          | 0.3414               | 0.1732           | 0.3850             | -0.0437              |
| weighted_f1       | 0.3591               | 0.1656           | 0.4007             | -0.0416              |
| unknown_rate      | 0.0000               | 0.0000           | 0.0000             | 0.0000               |

### before / test / 3Class Baseline / 三分类 Baseline

| Metric            | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
| ----------------- | -------------------- | ---------------- | ------------------ | -------------------- |
| accuracy          | 0.6401               | 0.3532           | 0.6505             | -0.0104              |
| balanced_accuracy | 0.5000               | 0.3429           | 0.5143             | -0.0143              |
| macro_f1          | 0.3903               | 0.1896           | 0.4200             | -0.0298              |
| weighted_f1       | 0.4996               | 0.1966           | 0.5227             | -0.0231              |
| unknown_rate      | 0.0000               | 0.0000           | 0.0000             | 0.0000               |

### before / test / 3Class Causal / 三分类 Causal

| Metric            | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
| ----------------- | -------------------- | ---------------- | ------------------ | -------------------- |
| accuracy          | 0.6401               | 0.3532           | 0.6505             | -0.0104              |
| balanced_accuracy | 0.5000               | 0.3429           | 0.5143             | -0.0143              |
| macro_f1          | 0.3903               | 0.1896           | 0.4200             | -0.0298              |
| weighted_f1       | 0.4996               | 0.1966           | 0.5227             | -0.0231              |
| unknown_rate      | 0.0000               | 0.0000           | 0.0000             | 0.0000               |

### after / val / 3Class Baseline / 三分类 Baseline

| Metric            | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
| ----------------- | -------------------- | ---------------- | ------------------ | -------------------- |
| accuracy          | 0.5209               | 0.2998           | 0.5406             | -0.0197              |
| balanced_accuracy | 0.5000               | 0.3473           | 0.5209             | -0.0209              |
| macro_f1          | 0.3414               | 0.1732           | 0.3850             | -0.0437              |
| weighted_f1       | 0.3591               | 0.1656           | 0.4007             | -0.0416              |
| unknown_rate      | 0.0000               | 0.0000           | 0.0000             | 0.0000               |

### after / val / 3Class Causal / 三分类 Causal

| Metric            | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
| ----------------- | -------------------- | ---------------- | ------------------ | -------------------- |
| accuracy          | 0.5209               | 0.2943           | 0.5350             | -0.0141              |
| balanced_accuracy | 0.5000               | 0.3428           | 0.5142             | -0.0142              |
| macro_f1          | 0.3414               | 0.1645           | 0.3718             | -0.0305              |
| weighted_f1       | 0.3591               | 0.1549           | 0.3892             | -0.0302              |
| unknown_rate      | 0.0000               | 0.0000           | 0.0000             | 0.0000               |

### after / test / 3Class Baseline / 三分类 Baseline

| Metric            | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
| ----------------- | -------------------- | ---------------- | ------------------ | -------------------- |
| accuracy          | 0.6401               | 0.3532           | 0.6505             | -0.0104              |
| balanced_accuracy | 0.5000               | 0.3429           | 0.5143             | -0.0143              |
| macro_f1          | 0.3903               | 0.1896           | 0.4200             | -0.0298              |
| weighted_f1       | 0.4996               | 0.1966           | 0.5227             | -0.0231              |
| unknown_rate      | 0.0000               | 0.0000           | 0.0000             | 0.0000               |

### after / test / 3Class Causal / 三分类 Causal

| Metric            | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |
| ----------------- | -------------------- | ---------------- | ------------------ | -------------------- |
| accuracy          | 0.6401               | 0.3532           | 0.6505             | -0.0104              |
| balanced_accuracy | 0.5000               | 0.3429           | 0.5143             | -0.0143              |
| macro_f1          | 0.3903               | 0.1896           | 0.4200             | -0.0298              |
| weighted_f1       | 0.4996               | 0.1966           | 0.5227             | -0.0231              |
| unknown_rate      | 0.0000               | 0.0000           | 0.0000             | 0.0000               |

## 3Class Baseline vs 3Class Causal / 三分类 Baseline 与 Causal 对比

### before / val

| Metric            | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
| ----------------- | ------------- | ----------- | ---------------------- |
| accuracy          | 0.2998        | 0.2998      | 0.0000                 |
| balanced_accuracy | 0.3473        | 0.3473      | 0.0000                 |
| macro_f1          | 0.1732        | 0.1732      | 0.0000                 |
| weighted_f1       | 0.1656        | 0.1656      | 0.0000                 |
| unknown_rate      | 0.0000        | 0.0000      | 0.0000                 |

Folded 2Class / 三分类折叠成二分类

| Metric                    | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
| ------------------------- | ------------- | ----------- | ---------------------- |
| folded::accuracy          | 0.5406        | 0.5406      | 0.0000                 |
| folded::balanced_accuracy | 0.5209        | 0.5209      | 0.0000                 |
| folded::macro_f1          | 0.3850        | 0.3850      | 0.0000                 |
| folded::weighted_f1       | 0.4007        | 0.4007      | 0.0000                 |
| folded::unknown_rate      | 0.0000        | 0.0000      | 0.0000                 |

### before / test

| Metric            | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
| ----------------- | ------------- | ----------- | ---------------------- |
| accuracy          | 0.3532        | 0.3532      | 0.0000                 |
| balanced_accuracy | 0.3429        | 0.3429      | 0.0000                 |
| macro_f1          | 0.1896        | 0.1896      | 0.0000                 |
| weighted_f1       | 0.1966        | 0.1966      | 0.0000                 |
| unknown_rate      | 0.0000        | 0.0000      | 0.0000                 |

Folded 2Class / 三分类折叠成二分类

| Metric                    | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
| ------------------------- | ------------- | ----------- | ---------------------- |
| folded::accuracy          | 0.6505        | 0.6505      | 0.0000                 |
| folded::balanced_accuracy | 0.5143        | 0.5143      | 0.0000                 |
| folded::macro_f1          | 0.4200        | 0.4200      | 0.0000                 |
| folded::weighted_f1       | 0.5227        | 0.5227      | 0.0000                 |
| folded::unknown_rate      | 0.0000        | 0.0000      | 0.0000                 |

### after / val

| Metric            | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
| ----------------- | ------------- | ----------- | ---------------------- |
| accuracy          | 0.2998        | 0.2943      | -0.0056                |
| balanced_accuracy | 0.3473        | 0.3428      | -0.0044                |
| macro_f1          | 0.1732        | 0.1645      | -0.0087                |
| weighted_f1       | 0.1656        | 0.1549      | -0.0107                |
| unknown_rate      | 0.0000        | 0.0000      | 0.0000                 |

Folded 2Class / 三分类折叠成二分类

| Metric                    | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
| ------------------------- | ------------- | ----------- | ---------------------- |
| folded::accuracy          | 0.5406        | 0.5350      | -0.0056                |
| folded::balanced_accuracy | 0.5209        | 0.5142      | -0.0067                |
| folded::macro_f1          | 0.3850        | 0.3718      | -0.0132                |
| folded::weighted_f1       | 0.4007        | 0.3892      | -0.0114                |
| folded::unknown_rate      | 0.0000        | 0.0000      | 0.0000                 |

### after / test

| Metric            | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
| ----------------- | ------------- | ----------- | ---------------------- |
| accuracy          | 0.3532        | 0.3532      | 0.0000                 |
| balanced_accuracy | 0.3429        | 0.3429      | 0.0000                 |
| macro_f1          | 0.1896        | 0.1896      | 0.0000                 |
| weighted_f1       | 0.1966        | 0.1966      | 0.0000                 |
| unknown_rate      | 0.0000        | 0.0000      | 0.0000                 |

Folded 2Class / 三分类折叠成二分类

| Metric                    | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |
| ------------------------- | ------------- | ----------- | ---------------------- |
| folded::accuracy          | 0.6505        | 0.6505      | 0.0000                 |
| folded::balanced_accuracy | 0.5143        | 0.5143      | 0.0000                 |
| folded::macro_f1          | 0.4200        | 0.4200      | 0.0000                 |
| folded::weighted_f1       | 0.5227        | 0.5227      | 0.0000                 |
| folded::unknown_rate      | 0.0000        | 0.0000      | 0.0000                 |
