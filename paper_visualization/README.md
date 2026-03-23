# Paper Visualization

这个目录专门服务论文写作，不修改训练与评估主线代码。

目标：
- 统一整理 `data1 official_cv` 与 `data2 transfer` 的实验结果
- 为论文生成高质量图表
- 提供一个可以一路跑到底的 notebook
- 提供图注、图号和正文引用建议

当前约定：
- `cache/`：中间整理后的 tidy 数据
- `exports/`：导出的 PNG / PDF / SVG / CSV
- `prepare_paper_data.py`：整理所有实验 summary、per-class 指标和预测明细
- `plot_paper_figures.py`：生成论文图表
- `paper_figures.ipynb`：从数据整理到绘图的一站式 notebook
- `FIGURE_CAPTIONS.md`：图注、图号和正文放置建议

推荐顺序：

```bash
export PROJECT_OUTPUT_ROOT="$(pwd)/outputs"
python paper_visualization/prepare_paper_data.py
python paper_visualization/plot_paper_figures.py
python paper_visualization/make_notebook.py
```

如果已经生成 notebook，也可以直接打开：

`paper_visualization/paper_figures.ipynb`

建议同时参考：

`paper_visualization/FIGURE_CAPTIONS.md`
