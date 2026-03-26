#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPO_ROOT / "paper_visualization" / "exports"


METHODS = [
    {
        "name": "Prompt-Qwen",
        "color": "#8C8C8C",
        "task1": {"macro_f1": 0.3914, "macro_f1_std": 0.0587, "bal_acc": 0.5021, "bal_acc_std": 0.0460},
        "task3": {"macro_f1": 0.4073, "macro_f1_std": 0.0511, "bal_acc": 0.4586, "bal_acc_std": 0.0286},
    },
    {
        "name": "Tri-LF",
        "color": "#3A7CA5",
        "task1": {"macro_f1": 0.7751, "macro_f1_std": 0.0522, "bal_acc": 0.7786, "bal_acc_std": 0.0531},
        "task3": {"macro_f1": 0.4236, "macro_f1_std": 0.1110, "bal_acc": 0.4973, "bal_acc_std": 0.0954},
    },
    {
        "name": "MCSQ-Net",
        "color": "#2A9D8F",
        "task1": {"macro_f1": 0.8547, "macro_f1_std": 0.0955, "bal_acc": 0.8556, "bal_acc_std": 0.0973},
        "task3": {"macro_f1": 0.6067, "macro_f1_std": 0.0351, "bal_acc": 0.6359, "bal_acc_std": 0.0158},
    },
]


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
            "grid.linewidth": 0.6,
            "axes.facecolor": "#FCFCFC",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
        }
    )


def ensure_dirs() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def compute_ylim(metric_key: str, std_key: str, floor: float) -> tuple[float, float]:
    values = []
    for method in METHODS:
        values.extend(
            [
                method["task1"][metric_key] - method["task1"][std_key],
                method["task1"][metric_key] + method["task1"][std_key],
                method["task3"][metric_key] - method["task3"][std_key],
                method["task3"][metric_key] + method["task3"][std_key],
            ]
        )
    lower = max(0.0, min(values) - 0.035)
    upper = min(1.0, max(values) + 0.035)
    return (min(lower, floor), upper)


def draw_panel(ax: plt.Axes, metric_key: str, std_key: str, title: str, ylim: tuple[float, float]) -> None:
    x_source, x_transfer = 0.0, 1.0

    ax.axvspan(0.82, 1.18, color="#F4EDE1", alpha=0.85, zorder=0)
    ax.axvline(0.5, color="#C9C9C9", linewidth=0.8, linestyle="--", zorder=0)

    for idx, method in enumerate(METHODS):
        source_mean = method["task1"][metric_key]
        source_std = method["task1"][std_key]
        transfer_mean = method["task3"][metric_key]
        transfer_std = method["task3"][std_key]
        color = method["color"]
        delta = transfer_mean - source_mean

        ax.plot(
            [x_source, x_transfer],
            [source_mean, transfer_mean],
            color=color,
            linewidth=2.2,
            alpha=0.95,
            zorder=2,
        )
        ax.scatter([x_source, x_transfer], [source_mean, transfer_mean], s=54, color=color, zorder=3)
        ax.errorbar(
            [x_source, x_transfer],
            [source_mean, transfer_mean],
            yerr=[source_std, transfer_std],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=3.5,
            zorder=1,
        )

        delta_text = f"$\\Delta$={delta:+.3f}"
        y_mid = (source_mean + transfer_mean) / 2
        x_mid = 0.48 + 0.02 * idx
        ax.text(
            x_mid,
            y_mid + (0.018 - idx * 0.012),
            delta_text,
            color=color,
            fontsize=8.5,
            ha="center",
            va="center",
        )

    ax.set_xlim(-0.12, 1.46)
    ax.set_ylim(*ylim)
    ax.set_xticks([x_source, x_transfer], ["Task-1\nSource held-out", "Task-3\nExternal transfer"])
    ax.set_ylabel(title)
    ax.set_title(title, loc="left", pad=8, fontweight="bold")
    ax.grid(axis="y", linestyle="-", alpha=0.55)
    ax.grid(axis="x", visible=False)


def build_figure() -> Path:
    set_style()
    ensure_dirs()

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 7.0), sharex=False)
    fig.patch.set_facecolor("white")

    draw_panel(axes[0], "macro_f1", "macro_f1_std", "Macro-F1", compute_ylim("macro_f1", "macro_f1_std", 0.28))
    draw_panel(axes[1], "bal_acc", "bal_acc_std", "Balanced Accuracy", compute_ylim("bal_acc", "bal_acc_std", 0.32))

    axes[0].text(
        -0.12,
        0.935,
        "(a)",
        transform=axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )
    axes[1].text(
        -0.12,
        0.935,
        "(b)",
        transform=axes[1].transAxes,
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(
        "External Transfer Performance and Cross-Domain Degradation",
        y=0.982,
        fontsize=12,
        fontweight="bold",
    )
    legend_handles = [
        Line2D([0], [0], color=method["color"], linewidth=2.4, marker="o", markersize=5.5, label=method["name"])
        for method in METHODS
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.8, 0.98),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.2,
        borderaxespad=0.0,
    )
    fig.text(
        0.02,
        0.015,
        "Mean +/- std over 3 folds. Task-3 introduces simultaneous device, process, and distribution shifts.",
        fontsize=8.5,
        color="#4F4F4F",
    )

    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.95))
    out_path = EXPORT_DIR / "fig_task3_transfer_summary_legend"
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".pdf")


if __name__ == "__main__":
    saved = build_figure()
    print(f"Saved figure to: {saved}")
