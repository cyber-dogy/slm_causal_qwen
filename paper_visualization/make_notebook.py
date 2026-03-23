#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "paper_visualization/paper_figures.ipynb"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md_cell(
            "# Paper Figures Notebook\n\n"
            "这个 notebook 用于论文图表的一键整理与导出。\n\n"
            "建议顺序：\n"
            "1. 重新整理实验数据\n"
            "2. 生成所有论文图\n"
            "3. 预览关键图表与表格\n"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import subprocess\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display, Markdown\n\n"
            "REPO_ROOT = Path.cwd().resolve().parents[0] if (Path.cwd().name == 'paper_visualization') else Path.cwd().resolve()\n"
            "if not (REPO_ROOT / 'paper_visualization').exists():\n"
            "    REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else REPO_ROOT\n"
            "PV_ROOT = REPO_ROOT / 'paper_visualization'\n"
            "CACHE_DIR = PV_ROOT / 'cache'\n"
            "EXPORT_DIR = PV_ROOT / 'exports'\n"
            "print('PV_ROOT =', PV_ROOT)\n"
        ),
        md_cell("## 1. 重新整理实验数据"),
        code_cell(
            "subprocess.run(['python', str(PV_ROOT / 'prepare_paper_data.py')], check=True)\n"
            "summary = json.loads((CACHE_DIR / 'prepare_summary.json').read_text(encoding='utf-8'))\n"
            "summary\n"
        ),
        md_cell("## 2. 生成论文图表"),
        code_cell(
            "subprocess.run(['python', str(PV_ROOT / 'plot_paper_figures.py')], check=True)\n"
            "plot_summary = json.loads((EXPORT_DIR / 'plot_summary.json').read_text(encoding='utf-8'))\n"
            "plot_summary\n"
        ),
        md_cell("## 3. 查看关键主结果表"),
        code_cell(
            "data1_main = pd.read_csv(EXPORT_DIR / 'table_data1_main_results.csv')\n"
            "data2_main = pd.read_csv(EXPORT_DIR / 'table_data2_transfer_results.csv')\n"
            "display(Markdown('### Data1 Main Results'))\n"
            "display(data1_main[['paper_label','benchmark','task_mode','split','stage','macro_f1_mean','balanced_accuracy_mean']].sort_values('macro_f1_mean', ascending=False))\n"
            "display(Markdown('### Data2 Transfer Results'))\n"
            "display(data2_main[['paper_label','benchmark','task_mode','split','stage','macro_f1_mean','balanced_accuracy_mean']].sort_values('macro_f1_mean', ascending=False))\n"
        ),
        md_cell("## 4. 预览关键图表"),
        code_cell(
            "figure_names = [\n"
            "    'fig01_data1_main_results.png',\n"
            "    'fig02_qwen_input_ablation.png',\n"
            "    'fig03_causal_gain.png',\n"
            "    'fig04_last2_before_after.png',\n"
            "    'fig05_data2_transfer_best.png',\n"
            "    'fig06_per_class_recall_heatmap.png',\n"
            "    'fig07_protocol_distribution.png',\n"
            "]\n"
            "for name in figure_names:\n"
            "    path = EXPORT_DIR / name\n"
            "    if path.exists():\n"
            "        display(Markdown(f'### {name}'))\n"
            "        display(Image(filename=str(path)))\n"
        ),
        md_cell("## 5. 查看 Qwen 输入消融表"),
        code_cell(
            "qwen_ablation = pd.read_csv(CACHE_DIR / 'qwen_input_ablation_3class.csv')\n"
            "qwen_ablation[['experiment_name','input_mode','method','macro_f1_mean','balanced_accuracy_mean']].sort_values('macro_f1_mean', ascending=False)\n"
        ),
        md_cell(
            "## 6. 后续建议\n\n"
            "- 若新增实验，只需重新运行本 notebook 前两步即可刷新图表。\n"
            "- 若要改论文配色或字体，直接编辑 `plot_paper_figures.py`。\n"
            "- 若要新增 figure，建议先在 `prepare_paper_data.py` 中补齐 tidy 数据列。 \n"
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
