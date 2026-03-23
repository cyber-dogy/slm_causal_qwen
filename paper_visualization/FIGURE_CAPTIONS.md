# Figure Captions And Placement Guide

这个文档用于论文写作时直接复用图注、正文引用句和图表放置建议。

适用范围：
- [论文_新版.md](/home/gjw/code/Qwen-SLM/论文_新版.md)
- [paper_figures.ipynb](/home/gjw/code/Qwen-SLM/paper_visualization/paper_figures.ipynb)
- `paper_visualization/exports/` 下的自动生成图表

## 使用建议

- 若正文采用中文撰写，可直接使用下文中文图题与图注。
- 若期刊要求英文图注，可使用下文英文版本。
- 自动生成图主要覆盖“实验结果与协议说明”。
- 方法总框架图仍建议手工绘制。

---

## Figure 1

文件：
- [fig01_data1_main_results.png](/home/gjw/code/Qwen-SLM/paper_visualization/exports/fig01_data1_main_results.png)

中文图题：
- `data1 official_cv` 三分类主结果对比

英文图题：
- Main 3-class results on the `data1 official_cv` protocol

中文图注：
- 图 1 展示了 `data1 official_cv` 条件级三分类任务上的主结果对比。与 prompt-Qwen 和经典多模态基线相比，QwenVisFusion 在显式视觉融合后显著提升了性能；在此基础上加入表征级因果一致性约束后，模型进一步取得当前最佳结果。相反，直接解冻最后两个视觉 block 的真微调并未带来收益，反而出现明显退化。

英文图注：
- Figure 1 compares the main 3-class results on the `data1 official_cv` condition-heldout protocol. QwenVisFusion substantially improves over prompt-Qwen and classical multimodal baselines after explicit visual fusion, and the representation-level causal regularization further delivers the best overall result. In contrast, true fine-tuning by unfreezing the last two visual blocks causes clear degradation.

正文引用建议：
- “如图 1 所示，QwenVisFusion + causal 在 `data1 official_cv` 三分类任务上取得了最佳结果。”

建议放置位置：
- 第 `6.1` 节“三分类主结果”之后

---

## Figure 2

文件：
- [fig02_qwen_input_ablation.png](/home/gjw/code/Qwen-SLM/paper_visualization/exports/fig02_qwen_input_ablation.png)

中文图题：
- QwenVisFusion 输入模式消融实验

英文图题：
- Input ablation of QwenVisFusion

中文图注：
- 图 2 对 QwenVisFusion 在 `rgb_view1`、`ir_only`、`rgb_dual` 和 `triple_view` 输入下的三分类性能进行了消融比较。结果表明，`rgb_dual` 显著优于单 RGB 和 IR-only，而在当前架构下将 IR 与双 RGB 直接并入并未继续提升性能。基于最优输入结构 `rgb_dual`，引入表征级因果一致性约束后可进一步获得额外增益。

英文图注：
- Figure 2 presents the 3-class input ablation of QwenVisFusion under `rgb_view1`, `ir_only`, `rgb_dual`, and `triple_view`. The results show that `rgb_dual` substantially outperforms both single RGB and IR-only inputs, while directly adding IR to the dual-RGB setup does not further improve performance in the current architecture. Applying representation-level causal regularization on top of the best input structure (`rgb_dual`) leads to an additional gain.

正文引用建议：
- “如图 2 所示，本文方法的关键不在于简单增加输入模态数量，而在于选择与当前任务最匹配的显式融合结构。”

建议放置位置：
- 第 `6.3` 节之后

---

## Figure 3

文件：
- [fig03_causal_gain.png](/home/gjw/code/Qwen-SLM/paper_visualization/exports/fig03_causal_gain.png)

中文图题：
- 表征级因果一致性约束带来的性能增益

英文图题：
- Performance gains brought by representation-level causal regularization

中文图注：
- 图 3 比较了 QwenVisFusion baseline 与 causal 在 `data1` 验证集、`data1` 测试集以及 `data2 transfer_test` 上的性能差异。可以看到，因果一致性约束在主闭集协议和外部迁移压力测试中都带来一致的正向增益，说明该约束不仅改善源域条件级泛化，也提升了更强分布偏移下的鲁棒性。

英文图注：
- Figure 3 compares the baseline and causal variants of QwenVisFusion on the `data1` validation set, the `data1` test set, and the `data2 transfer_test`. The causal regularization consistently improves performance across both the source-side condition-heldout protocol and the external transfer stress test, indicating that it benefits not only condition generalization but also robustness under stronger distribution shifts.

正文引用建议：
- “图 3 进一步验证了表征级因果一致性约束的有效性，其提升不仅体现在 source-side 条件级泛化，也体现在 data2 的外部迁移测试上。”

建议放置位置：
- 第 `6.2` 节之后

---

## Figure 4

文件：
- [fig04_last2_before_after.png](/home/gjw/code/Qwen-SLM/paper_visualization/exports/fig04_last2_before_after.png)

中文图题：
- 解冻最后两个视觉 block 的 before/after 对比

英文图题：
- Before/after comparison of unfreezing the last two visual blocks

中文图注：
- 图 4 展示了基于 frozen-head checkpoint 的 `last2blocks` 真微调结果。无论是在 `data1 official_cv` 还是在 `data2 transfer_test` 上，解冻最后两个视觉 block 后的性能均低于解冻前的 frozen-head 版本，表明在当前小样本工业场景下，激进真微调会引发表征漂移与灾难性遗忘风险。

英文图注：
- Figure 4 shows the before/after comparison of the `last2blocks` true fine-tuning initialized from the frozen-head checkpoints. On both `data1 official_cv` and `data2 transfer_test`, the performance after unfreezing the last two visual blocks is lower than the frozen-head counterpart, indicating that aggressive true fine-tuning introduces representation drift and catastrophic forgetting in the current small-sample industrial setting.

正文引用建议：
- “图 4 说明，当前最优路线并不是继续激进解冻视觉塔，而是保持视觉编码稳定并通过轻量判别式头进行适配。”

建议放置位置：
- 第 `6.8` 节之后

---

## Figure 5

文件：
- [fig05_data2_transfer_best.png](/home/gjw/code/Qwen-SLM/paper_visualization/exports/fig05_data2_transfer_best.png)

中文图题：
- `data2 transfer_test` 外部迁移最佳结果对比

英文图题：
- Best model comparison on `data2 transfer_test`

中文图注：
- 图 5 比较了三大家族模型在 `data2 transfer_test` 上的最佳结果。虽然 `data2` 相比 `data1 official_cv` 具有更强的未见模型、未见工艺与复合分布偏移压力，但 QwenVisFusion + causal 仍取得了最优的三分类迁移性能，说明其不仅适用于源域闭集条件级泛化，也具有更强的外部迁移鲁棒性。

英文图注：
- Figure 5 compares the best-performing models from the three model families on `data2 transfer_test`. Although `data2` introduces substantially harder shifts due to unseen models, unseen processes, and composite distribution changes, QwenVisFusion + causal still achieves the best 3-class transfer performance, suggesting that its benefit extends beyond source-side condition-heldout generalization to external transfer robustness.

正文引用建议：
- “如图 5 所示，QwenVisFusion + causal 在 `data2` 外部迁移压力测试中仍保持了最优的三分类性能。”

建议放置位置：
- 第 `6.7` 节之后

---

## Figure 6

文件：
- [fig06_per_class_recall_heatmap.png](/home/gjw/code/Qwen-SLM/paper_visualization/exports/fig06_per_class_recall_heatmap.png)

中文图题：
- `data1 official_cv` 测试集上的分类别召回热图

英文图题：
- Per-class recall heatmap on the `data1 official_cv` test set

中文图注：
- 图 6 展示了代表性模型在 `data1 official_cv` 测试集上的分类别召回表现。可以看到，prompt-Qwen 在三类上均明显偏弱，而 QwenVisFusion + causal 在 `HEW` 和 `LEL` 上都取得了较高召回，同时在 `normal` 上保持稳定表现，说明其性能提升并非仅来自单一类别。

英文图注：
- Figure 6 shows the per-class recall of representative models on the `data1 official_cv` test set. Prompt-Qwen is weak across all three classes, whereas QwenVisFusion + causal achieves strong recall on both `HEW` and `LEL` while maintaining stable performance on `normal`, indicating that the improvement is not driven by a single class only.

正文引用建议：
- “图 6 表明，QwenVisFusion + causal 的增益并不是某一类别上的偶然提升，而是在三类上形成了较均衡的判别能力。”

建议放置位置：
- 第 `6.2` 节或 `6.6` 节之后

---

## Figure 7

文件：
- [fig07_protocol_distribution.png](/home/gjw/code/Qwen-SLM/paper_visualization/exports/fig07_protocol_distribution.png)

中文图题：
- `data1` 与 `data2` 协议和分布概览

英文图题：
- Protocol and distribution overview of `data1` and `data2`

中文图注：
- 图 7 展示了 `data1` 与 `data2` 的标签分布对比以及 `data2` 的角色分布。该图用于说明两点：其一，`data1` 与 `data2` 在样本规模和标签构成上并不完全一致；其二，`data2` 包含更多复合角色与分布偏移，因此应被解释为外部迁移压力测试，而非与 `data1 official_cv` 同难度的闭集验证。

英文图注：
- Figure 7 illustrates the label distributions of `data1` and `data2`, together with the role distribution in `data2`. This figure serves two purposes: first, `data1` and `data2` do not share identical sample scales or label compositions; second, `data2` contains richer composite roles and distribution shifts, and should therefore be interpreted as an external transfer stress test rather than a closed-set validation protocol of the same difficulty as `data1 official_cv`.

正文引用建议：
- “如图 7 所示，`data2` 的角色与样本分布明显更复杂，因此本文将其定位为 external transfer stress test，而非主选模集。”

建议放置位置：
- 第 `3` 节协议说明之后

---

## 建议补画的手工图

当前自动生成图覆盖了主要实验结果，但仍建议补一张手工绘制的“方法框架图”：

- 左侧：
  `rgb_view1_after`、`rgb_view2_after`
- 中部：
  Qwen2-VL vision tower
- 后续：
  MLP fusion head
- 顶部：
  classification loss
- 底部：
  same-label different-condition causal consistency loss

这张图更适合作为论文中的第一张方法图。
