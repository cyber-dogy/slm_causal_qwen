# Data1 Split Description / Data1 划分说明

Scope / 协议范围: `official_cv`

Protocol Root / 协议根目录: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol`

## Protocol Rules / 划分规则

- Split unit: `condition_uid` rather than single samples / 划分单位是 `condition_uid`，不是单样本。
- Within each fold, train/val/test are condition-disjoint / 每一折内部的 train/val/test 在 `condition_uid` 上严格不重叠。
- `train` is class-balanced by manifest-level upsampling to max class count / `train` 通过 manifest 级上采样到最大类样本数实现类别均衡。
- `val/test` keep the natural sample counts inside the selected conditions / `val/test` 保持所选 condition 内的自然样本数，不做重采样。
- To reproduce Qwen training exactly in ResNet or other classical models, reuse the same `train_manifest.jsonl` rows as-is, including duplicated rows from upsampling / 如果想和 Qwen 训练分布严格对齐，ResNet 等经典网络应直接复用相同的 `train_manifest.jsonl` 行，包括上采样带来的重复行。
- If you deduplicate train rows and resample in your own dataloader, the condition split is the same but the training distribution is no longer identical to Qwen / 如果你对 train 去重后自己在 dataloader 中重采样，那么 condition 划分仍相同，但训练分布将不再与 Qwen 完全一致。

## Build Config / 构建配置

- Total samples / 总样本数: `289`
- Total conditions / condition 总数: `22`
- Official folds / 官方折数: `3`
- Official val ratio / 官方 val 比例: `0.25`
- Train balancing / 训练集均衡: `True`
- Balance mode / 均衡方式: `upsample_to_max`
- Seed / 随机种子: `42`

## Condition Catalog / 全部 Condition 清单

| Condition | Label3C | Label2C | Role | Raw Samples |
|-----------|---------|---------|------|-------------|
| `data1__Base-01` | `normal` | `normal` | `source_train` | 14 |
| `data1__Base-02` | `normal` | `normal` | `source_repeat` | 10 |
| `data1__Equiv-01` | `normal` | `normal` | `causal_equiv_test` | 13 |
| `data1__Equiv-02` | `normal` | `normal` | `causal_equiv_test` | 11 |
| `data1__Gen-01` | `normal` | `normal` | `unseen_strategy_shift` | 10 |
| `data1__Gen-02` | `normal` | `normal` | `unseen_texture_shift` | 13 |
| `data1__Gen-03` | `normal` | `normal` | `unseen_texture_shift` | 12 |
| `data1__High-01` | `normal` | `normal` | `causal_intervention_high` | 10 |
| `data1__High-02` | `HEW` | `abnormal` | `causal_intervention_high` | 12 |
| `data1__High-03` | `HEW` | `abnormal` | `causal_intervention_high` | 13 |
| `data1__High-04` | `HEW` | `abnormal` | `causal_intervention_high` | 11 |
| `data1__High-05` | `HEW` | `abnormal` | `unseen_strategy_shift` | 16 |
| `data1__High-06` | `HEW` | `abnormal` | `unseen_composite_shift` | 15 |
| `data1__High-07` | `HEW` | `abnormal` | `unseen_texture_shift` | 7 |
| `data1__High-08` | `HEW` | `abnormal` | `unseen_texture_shift` | 12 |
| `data1__Low-01` | `normal` | `normal` | `causal_intervention_low` | 11 |
| `data1__Low-02` | `LEL` | `abnormal` | `causal_intervention_low` | 15 |
| `data1__Low-03` | `LEL` | `abnormal` | `causal_intervention_low` | 25 |
| `data1__Low-04` | `LEL` | `abnormal` | `causal_intervention_low` | 22 |
| `data1__Low-05` | `LEL` | `abnormal` | `unseen_param_shift` | 16 |
| `data1__Low-06` | `LEL` | `abnormal` | `unseen_param_shift` | 11 |
| `data1__Low-07` | `LEL` | `abnormal` | `unseen_composite_shift` | 10 |

## Fold Details / 各折明细

### fold_01

Root / 目录: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_01`

| Split | Manifest Rows | Unique Original IDs | Duplicated Rows | Unique Conditions | Label3C | Role Conditions |
|-------|---------------|---------------------|-----------------|-------------------|---------|-----------------|
| train | 168 | 155 | 13 | 11 | `HEW=56, LEL=56, normal=56` | `causal_equiv_test=2, causal_intervention_high=3, causal_intervention_low=2, source_repeat=1, unseen_param_shift=1, unseen_strategy_shift=1, unseen_texture_shift=1` |
| val | 39 | 39 | 0 | 4 | `HEW=7, LEL=10, normal=22` | `causal_intervention_high=1, unseen_composite_shift=1, unseen_texture_shift=2` |
| test | 95 | 95 | 0 | 7 | `HEW=27, LEL=33, normal=35` | `causal_intervention_low=2, source_train=1, unseen_composite_shift=1, unseen_param_shift=1, unseen_strategy_shift=1, unseen_texture_shift=1` |

#### train

Manifest / 清单: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_01/train_manifest.jsonl`

| Condition | Label3C | Role | Manifest Rows | Unique Original IDs | Duplicated Rows |
|-----------|---------|------|---------------|---------------------|-----------------|
| `data1__Base-02` | `normal` | `source_repeat` | 13 | 10 | 3 |
| `data1__Equiv-01` | `normal` | `causal_equiv_test` | 14 | 13 | 1 |
| `data1__Equiv-02` | `normal` | `causal_equiv_test` | 13 | 11 | 2 |
| `data1__Gen-02` | `normal` | `unseen_texture_shift` | 16 | 13 | 3 |
| `data1__High-02` | `HEW` | `causal_intervention_high` | 12 | 12 | 0 |
| `data1__High-03` | `HEW` | `causal_intervention_high` | 13 | 13 | 0 |
| `data1__High-04` | `HEW` | `causal_intervention_high` | 13 | 11 | 2 |
| `data1__High-05` | `HEW` | `unseen_strategy_shift` | 18 | 16 | 2 |
| `data1__Low-02` | `LEL` | `causal_intervention_low` | 15 | 15 | 0 |
| `data1__Low-03` | `LEL` | `causal_intervention_low` | 25 | 25 | 0 |
| `data1__Low-05` | `LEL` | `unseen_param_shift` | 16 | 16 | 0 |

#### val

Manifest / 清单: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_01/val_manifest.jsonl`

| Condition | Label3C | Role | Manifest Rows | Unique Original IDs | Duplicated Rows |
|-----------|---------|------|---------------|---------------------|-----------------|
| `data1__Gen-03` | `normal` | `unseen_texture_shift` | 12 | 12 | 0 |
| `data1__High-01` | `normal` | `causal_intervention_high` | 10 | 10 | 0 |
| `data1__High-07` | `HEW` | `unseen_texture_shift` | 7 | 7 | 0 |
| `data1__Low-07` | `LEL` | `unseen_composite_shift` | 10 | 10 | 0 |

#### test

Manifest / 清单: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_01/test_manifest.jsonl`

| Condition | Label3C | Role | Manifest Rows | Unique Original IDs | Duplicated Rows |
|-----------|---------|------|---------------|---------------------|-----------------|
| `data1__Base-01` | `normal` | `source_train` | 14 | 14 | 0 |
| `data1__Gen-01` | `normal` | `unseen_strategy_shift` | 10 | 10 | 0 |
| `data1__High-06` | `HEW` | `unseen_composite_shift` | 15 | 15 | 0 |
| `data1__High-08` | `HEW` | `unseen_texture_shift` | 12 | 12 | 0 |
| `data1__Low-01` | `normal` | `causal_intervention_low` | 11 | 11 | 0 |
| `data1__Low-04` | `LEL` | `causal_intervention_low` | 22 | 22 | 0 |
| `data1__Low-06` | `LEL` | `unseen_param_shift` | 11 | 11 | 0 |

### fold_02

Root / 目录: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_02`

| Split | Manifest Rows | Unique Original IDs | Duplicated Rows | Unique Conditions | Label3C | Role Conditions |
|-------|---------------|---------------------|-----------------|-------------------|---------|-----------------|
| train | 174 | 145 | 29 | 10 | `HEW=58, LEL=58, normal=58` | `causal_intervention_high=1, causal_intervention_low=3, source_train=1, unseen_param_shift=1, unseen_strategy_shift=2, unseen_texture_shift=2` |
| val | 46 | 46 | 0 | 4 | `HEW=15, LEL=10, normal=21` | `causal_equiv_test=1, source_repeat=1, unseen_composite_shift=2` |
| test | 98 | 98 | 0 | 8 | `HEW=32, LEL=31, normal=35` | `causal_equiv_test=1, causal_intervention_high=3, causal_intervention_low=1, unseen_param_shift=1, unseen_texture_shift=2` |

#### train

Manifest / 清单: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_02/train_manifest.jsonl`

| Condition | Label3C | Role | Manifest Rows | Unique Original IDs | Duplicated Rows |
|-----------|---------|------|---------------|---------------------|-----------------|
| `data1__Base-01` | `normal` | `source_train` | 17 | 14 | 3 |
| `data1__Gen-01` | `normal` | `unseen_strategy_shift` | 12 | 10 | 2 |
| `data1__Gen-02` | `normal` | `unseen_texture_shift` | 17 | 13 | 4 |
| `data1__High-04` | `HEW` | `causal_intervention_high` | 18 | 11 | 7 |
| `data1__High-05` | `HEW` | `unseen_strategy_shift` | 24 | 16 | 8 |
| `data1__High-08` | `HEW` | `unseen_texture_shift` | 16 | 12 | 4 |
| `data1__Low-01` | `normal` | `causal_intervention_low` | 12 | 11 | 1 |
| `data1__Low-03` | `LEL` | `causal_intervention_low` | 25 | 25 | 0 |
| `data1__Low-04` | `LEL` | `causal_intervention_low` | 22 | 22 | 0 |
| `data1__Low-06` | `LEL` | `unseen_param_shift` | 11 | 11 | 0 |

#### val

Manifest / 清单: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_02/val_manifest.jsonl`

| Condition | Label3C | Role | Manifest Rows | Unique Original IDs | Duplicated Rows |
|-----------|---------|------|---------------|---------------------|-----------------|
| `data1__Base-02` | `normal` | `source_repeat` | 10 | 10 | 0 |
| `data1__Equiv-02` | `normal` | `causal_equiv_test` | 11 | 11 | 0 |
| `data1__High-06` | `HEW` | `unseen_composite_shift` | 15 | 15 | 0 |
| `data1__Low-07` | `LEL` | `unseen_composite_shift` | 10 | 10 | 0 |

#### test

Manifest / 清单: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_02/test_manifest.jsonl`

| Condition | Label3C | Role | Manifest Rows | Unique Original IDs | Duplicated Rows |
|-----------|---------|------|---------------|---------------------|-----------------|
| `data1__Equiv-01` | `normal` | `causal_equiv_test` | 13 | 13 | 0 |
| `data1__Gen-03` | `normal` | `unseen_texture_shift` | 12 | 12 | 0 |
| `data1__High-01` | `normal` | `causal_intervention_high` | 10 | 10 | 0 |
| `data1__High-02` | `HEW` | `causal_intervention_high` | 12 | 12 | 0 |
| `data1__High-03` | `HEW` | `causal_intervention_high` | 13 | 13 | 0 |
| `data1__High-07` | `HEW` | `unseen_texture_shift` | 7 | 7 | 0 |
| `data1__Low-02` | `LEL` | `causal_intervention_low` | 15 | 15 | 0 |
| `data1__Low-05` | `LEL` | `unseen_param_shift` | 16 | 16 | 0 |

### fold_03

Root / 目录: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_03`

| Split | Manifest Rows | Unique Original IDs | Duplicated Rows | Unique Conditions | Label3C | Role Conditions |
|-------|---------------|---------------------|-----------------|-------------------|---------|-----------------|
| train | 138 | 133 | 5 | 11 | `HEW=46, LEL=46, normal=46` | `causal_equiv_test=1, causal_intervention_high=2, causal_intervention_low=1, unseen_composite_shift=1, unseen_param_shift=2, unseen_strategy_shift=1, unseen_texture_shift=3` |
| val | 60 | 60 | 0 | 4 | `HEW=13, LEL=22, normal=25` | `causal_intervention_high=1, causal_intervention_low=2, source_train=1` |
| test | 96 | 96 | 0 | 7 | `HEW=27, LEL=35, normal=34` | `causal_equiv_test=1, causal_intervention_high=1, causal_intervention_low=1, source_repeat=1, unseen_composite_shift=1, unseen_strategy_shift=1, unseen_texture_shift=1` |

#### train

Manifest / 清单: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_03/train_manifest.jsonl`

| Condition | Label3C | Role | Manifest Rows | Unique Original IDs | Duplicated Rows |
|-----------|---------|------|---------------|---------------------|-----------------|
| `data1__Equiv-01` | `normal` | `causal_equiv_test` | 13 | 13 | 0 |
| `data1__Gen-01` | `normal` | `unseen_strategy_shift` | 11 | 10 | 1 |
| `data1__Gen-03` | `normal` | `unseen_texture_shift` | 12 | 12 | 0 |
| `data1__High-01` | `normal` | `causal_intervention_high` | 10 | 10 | 0 |
| `data1__High-02` | `HEW` | `causal_intervention_high` | 12 | 12 | 0 |
| `data1__High-06` | `HEW` | `unseen_composite_shift` | 15 | 15 | 0 |
| `data1__High-07` | `HEW` | `unseen_texture_shift` | 7 | 7 | 0 |
| `data1__High-08` | `HEW` | `unseen_texture_shift` | 12 | 12 | 0 |
| `data1__Low-02` | `LEL` | `causal_intervention_low` | 16 | 15 | 1 |
| `data1__Low-05` | `LEL` | `unseen_param_shift` | 18 | 16 | 2 |
| `data1__Low-06` | `LEL` | `unseen_param_shift` | 12 | 11 | 1 |

#### val

Manifest / 清单: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_03/val_manifest.jsonl`

| Condition | Label3C | Role | Manifest Rows | Unique Original IDs | Duplicated Rows |
|-----------|---------|------|---------------|---------------------|-----------------|
| `data1__Base-01` | `normal` | `source_train` | 14 | 14 | 0 |
| `data1__High-03` | `HEW` | `causal_intervention_high` | 13 | 13 | 0 |
| `data1__Low-01` | `normal` | `causal_intervention_low` | 11 | 11 | 0 |
| `data1__Low-04` | `LEL` | `causal_intervention_low` | 22 | 22 | 0 |

#### test

Manifest / 清单: `/home/gjw/code/SLM_data/processed_qwen_data1_protocol/official_cv/fold_03/test_manifest.jsonl`

| Condition | Label3C | Role | Manifest Rows | Unique Original IDs | Duplicated Rows |
|-----------|---------|------|---------------|---------------------|-----------------|
| `data1__Base-02` | `normal` | `source_repeat` | 10 | 10 | 0 |
| `data1__Equiv-02` | `normal` | `causal_equiv_test` | 11 | 11 | 0 |
| `data1__Gen-02` | `normal` | `unseen_texture_shift` | 13 | 13 | 0 |
| `data1__High-04` | `HEW` | `causal_intervention_high` | 11 | 11 | 0 |
| `data1__High-05` | `HEW` | `unseen_strategy_shift` | 16 | 16 | 0 |
| `data1__Low-03` | `LEL` | `causal_intervention_low` | 25 | 25 | 0 |
| `data1__Low-07` | `LEL` | `unseen_composite_shift` | 10 | 10 | 0 |

