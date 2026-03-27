#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPO_ROOT / "paper_visualization" / "exports"


METHODS = ["Prompt-Qwen", "Dual", "MCSQ-Net"]
METRICS = {
    "Balanced Accuracy": {
        "means": [0.5000, 0.8506, 0.5429],
        "stds": [0.0000, 0.0981, 0.0606],
        "color": "#7AA6DC",
        "hatch": "///",
    },
    "Macro-F1": {
        "means": [0.3903, 0.8588, 0.4707],
        "stds": [0.0023, 0.1011, 0.1145],
        "color": "#D98B5A",
        "hatch": "\\\\\\",
    },
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "axes.facecolor": "#FBFBFB",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )


def ensure_dirs() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_figure() -> Path:
    set_style()
    ensure_dirs()

    fig, ax = plt.subplots(figsize=(4.8, 3.15))
    fig.patch.set_facecolor("white")

    x = np.arange(len(METHODS))
    width = 0.34

    for idx, (metric_name, metric) in enumerate(METRICS.items()):
        offset = (-0.5 + idx) * width
        bars = ax.bar(
            x + offset,
            metric["means"],
            width=width,
            color=metric["color"],
            edgecolor="#4F4F4F",
            linewidth=0.7,
            hatch=metric["hatch"],
            label=metric_name,
            zorder=2,
        )
        ax.errorbar(
            x + offset,
            metric["means"],
            yerr=metric["stds"],
            fmt="none",
            ecolor="#4F4F4F",
            elinewidth=0.9,
            capsize=2.6,
            zorder=3,
        )
        for bar, value, std in zip(bars, metric["means"], metric["stds"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + std + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                color="#3F3F3F",
            )

    ax.set_xticks(x, METHODS)
    ax.set_ylim(0.30, 1.02)
    ax.set_ylabel("Score")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handlelength=1.6,
        borderaxespad=0.0,
    )
    ax.grid(axis="y", linestyle="-", alpha=0.55)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    out_path = EXPORT_DIR / "fig_task2_anomaly_bar"
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".pdf")


if __name__ == "__main__":
    saved = build_figure()
    print(f"Saved figure to: {saved}")
