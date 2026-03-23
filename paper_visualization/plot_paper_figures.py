#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "paper_visualization"
CACHE_DIR = OUTPUT_ROOT / "cache"
EXPORT_DIR = OUTPUT_ROOT / "exports"


COLORS = {
    "prompt": "#8C8C8C",
    "classic": "#3A7CA5",
    "qwen_base": "#E09F3E",
    "qwen_causal": "#2A9D8F",
    "qwen_last2": "#D1495B",
    "qwen_last2_causal": "#7A306C",
    "single": "#7C8DA6",
    "ir": "#C97C5D",
    "triple": "#6C9A8B",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "legend.frameon": False,
        }
    )


def ensure_dirs() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(EXPORT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(EXPORT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_inputs() -> Dict[str, pd.DataFrame]:
    return {
        "metrics": pd.read_csv(CACHE_DIR / "paper_metrics_all.csv"),
        "data1_focus": pd.read_csv(CACHE_DIR / "focus_data1_3class.csv"),
        "qwen_ablation": pd.read_csv(CACHE_DIR / "qwen_input_ablation_3class.csv"),
        "transfer_focus": pd.read_csv(CACHE_DIR / "focus_data2_transfer.csv"),
        "per_class": pd.read_csv(CACHE_DIR / "paper_per_class_selected.csv"),
    }


def canonical_label(row: pd.Series) -> str:
    exp = row["experiment_name"]
    family = row["family"]
    stage = row.get("stage", "after")
    if family == "prompt_qwen" and exp == "3class_baseline":
        return "Prompt-Qwen"
    if family == "classic" and exp == "resnet18_rgb_dual_mlp_concat_3class":
        return "Classic RGB-Dual"
    if exp in {"qwen_vis_rgb_dual_baseline_3class", "official_cv_3class_rgb_dual_baseline"}:
        return "QwenVis Baseline"
    if exp in {"qwen_vis_rgb_dual_causal_3class", "official_cv_3class_rgb_dual_causal"}:
        return "QwenVis Causal"
    if exp in {"qwen_vis_rgb_dual_last2blocks_baseline_3class", "official_cv_3class_rgb_dual_last2blocks_baseline"}:
        return "Last2 Baseline" if stage == "after" else "Frozen Before"
    if exp in {"qwen_vis_rgb_dual_last2blocks_causal_3class", "official_cv_3class_rgb_dual_last2blocks_causal"}:
        return "Last2 Causal" if stage == "after" else "Frozen Causal Before"
    if exp == "qwen_vis_rgb_view1_baseline_3class":
        return "RGBv1 Only"
    if exp == "qwen_vis_ir_only_baseline_3class":
        return "IR Only"
    if exp == "qwen_vis_triple_view_baseline_3class":
        return "RGB-Dual + IR"
    if exp == "qwen_vis_rgb_dual_baseline_3class":
        return "RGB-Dual"
    if exp == "qwen_vis_rgb_dual_causal_3class":
        return "RGB-Dual + Causal"
    if exp == "data1_official_cv_3class_baseline":
        return "Prompt-Qwen Transfer"
    if exp == "resnet18_triple_view_late_fusion_3class":
        return "Classic Transfer Best"
    return exp


def add_errorbars(ax: plt.Axes, df: pd.DataFrame, mean_col: str, std_col: str) -> None:
    patches = [patch for patch in ax.patches if patch.get_height() == patch.get_height()]
    limit = min(len(patches), len(df))
    for patch, (_, row) in zip(patches[:limit], df.iloc[:limit].iterrows()):
        ax.errorbar(
            x=patch.get_x() + patch.get_width() / 2,
            y=row[mean_col],
            yerr=row[std_col],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.2,
        )


def fig_data1_main_results(data1_focus: pd.DataFrame) -> None:
    df = data1_focus.copy()
    df = df[df["stage"] == "after"].copy()
    df["label"] = df.apply(canonical_label, axis=1)
    keep_order = [
        "Prompt-Qwen",
        "Classic RGB-Dual",
        "QwenVis Baseline",
        "QwenVis Causal",
        "Last2 Baseline",
        "Last2 Causal",
    ]
    df = df[df["label"].isin(keep_order)].copy()
    df["label"] = pd.Categorical(df["label"], categories=keep_order, ordered=True)
    df = df.sort_values("label")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    palette = {
        "Prompt-Qwen": COLORS["prompt"],
        "Classic RGB-Dual": COLORS["classic"],
        "QwenVis Baseline": COLORS["qwen_base"],
        "QwenVis Causal": COLORS["qwen_causal"],
        "Last2 Baseline": COLORS["qwen_last2"],
        "Last2 Causal": COLORS["qwen_last2_causal"],
    }
    for ax, metric, title in [
        (axes[0], "macro_f1_mean", "Macro-F1"),
        (axes[1], "balanced_accuracy_mean", "Balanced Accuracy"),
    ]:
        sns.barplot(data=df, x="label", y=metric, hue="label", dodge=False, palette=palette, legend=False, ax=ax)
        add_errorbars(ax, df, metric, metric.replace("_mean", "_std"))
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle("Data1 Official-CV 3-Class Main Results", fontsize=18, fontweight="bold")
    save_figure(fig, "fig01_data1_main_results")


def fig_qwen_input_ablation(qwen_ablation: pd.DataFrame) -> None:
    df = qwen_ablation.copy()
    df["label"] = df.apply(canonical_label, axis=1)
    order = ["RGBv1 Only", "IR Only", "RGB-Dual", "RGB-Dual + IR", "RGB-Dual + Causal"]
    df = df[df["label"].isin(order)].copy()
    df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
    df = df.sort_values("label")
    palette = {
        "RGBv1 Only": COLORS["single"],
        "IR Only": COLORS["ir"],
        "RGB-Dual": COLORS["qwen_base"],
        "RGB-Dual + IR": COLORS["triple"],
        "RGB-Dual + Causal": COLORS["qwen_causal"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    for ax, metric, title in [
        (axes[0], "macro_f1_mean", "Macro-F1"),
        (axes[1], "balanced_accuracy_mean", "Balanced Accuracy"),
    ]:
        sns.barplot(data=df, x="label", y=metric, hue="label", dodge=False, palette=palette, legend=False, ax=ax)
        add_errorbars(ax, df, metric, metric.replace("_mean", "_std"))
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle("QwenVisFusion 3-Class Input Ablation", fontsize=18, fontweight="bold")
    save_figure(fig, "fig02_qwen_input_ablation")


def fig_causal_gain(metrics: pd.DataFrame) -> None:
    rows: List[Dict[str, object]] = []
    for benchmark, split, exp_base, exp_causal in [
        ("data1_official_cv", "val", "qwen_vis_rgb_dual_baseline_3class", "qwen_vis_rgb_dual_causal_3class"),
        ("data1_official_cv", "test", "qwen_vis_rgb_dual_baseline_3class", "qwen_vis_rgb_dual_causal_3class"),
        ("data2_transfer", "transfer_test", "official_cv_3class_rgb_dual_baseline", "official_cv_3class_rgb_dual_causal"),
    ]:
        subset = metrics[(metrics["benchmark"] == benchmark) & (metrics["task_mode"] == "3class") & (metrics["split"] == split) & (metrics["stage"] == "after")]
        for exp in [exp_base, exp_causal]:
            row = subset[subset["experiment_name"] == exp]
            if row.empty:
                continue
            rec = row.iloc[0].to_dict()
            rec["label"] = "Causal" if "causal" in exp else "Baseline"
            rec["panel"] = f"{benchmark}:{split}"
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return
    panel_map = {
        "data1_official_cv:val": "Data1 Val",
        "data1_official_cv:test": "Data1 Test",
        "data2_transfer:transfer_test": "Data2 Transfer",
    }
    df["panel"] = df["panel"].map(panel_map)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    for ax, metric, title in [
        (axes[0], "macro_f1_mean", "Macro-F1"),
        (axes[1], "balanced_accuracy_mean", "Balanced Accuracy"),
    ]:
        sns.barplot(
            data=df,
            x="panel",
            y=metric,
            hue="label",
            palette={"Baseline": COLORS["qwen_base"], "Causal": COLORS["qwen_causal"]},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Effect of Representation-Level Causal Regularization", fontsize=18, fontweight="bold")
    save_figure(fig, "fig03_causal_gain")


def fig_last2_before_after(metrics: pd.DataFrame) -> None:
    rows: List[Dict[str, object]] = []
    for benchmark, split, experiments in [
        (
            "data1_official_cv",
            "test",
            [
                ("qwen_vis_rgb_dual_last2blocks_baseline_3class", "Last2 Baseline"),
                ("qwen_vis_rgb_dual_last2blocks_causal_3class", "Last2 Causal"),
            ],
        ),
        (
            "data2_transfer",
            "transfer_test",
            [
                ("official_cv_3class_rgb_dual_last2blocks_baseline", "Last2 Baseline"),
                ("official_cv_3class_rgb_dual_last2blocks_causal", "Last2 Causal"),
            ],
        ),
    ]:
        for exp, label in experiments:
            subset = metrics[
                (metrics["benchmark"] == benchmark)
                & (metrics["split"] == split)
                & (metrics["task_mode"] == "3class")
                & (metrics["experiment_name"] == exp)
                & (metrics["stage"].isin(["before", "after"]))
            ]
            for _, row in subset.iterrows():
                rows.append(
                    {
                        "panel": "Data1 Test" if benchmark == "data1_official_cv" else "Data2 Transfer",
                        "variant": label,
                        "stage": row["stage"].capitalize(),
                        "macro_f1_mean": row["macro_f1_mean"],
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    panel_order = ["Data1 Test", "Data2 Transfer"]
    color_map = {"Last2 Baseline": COLORS["qwen_last2"], "Last2 Causal": COLORS["qwen_last2_causal"]}
    for ax, panel in zip(axes, panel_order):
        sub = df[df["panel"] == panel]
        for variant in ["Last2 Baseline", "Last2 Causal"]:
            sub_var = sub[sub["variant"] == variant].set_index("stage").reindex(["Before", "After"])
            if sub_var.empty or sub_var["macro_f1_mean"].isna().any():
                continue
            ax.plot(["Before", "After"], sub_var["macro_f1_mean"], marker="o", linewidth=3, color=color_map[variant], label=variant)
        ax.set_title(panel)
        ax.set_xlabel("")
        ax.set_ylabel("Macro-F1")
        ax.legend()
    fig.suptitle("True Fine-Tuning Causes Degradation", fontsize=18, fontweight="bold")
    save_figure(fig, "fig04_last2_before_after")


def fig_transfer_family_best(transfer_focus: pd.DataFrame) -> None:
    df = transfer_focus.copy()
    best_rows = []
    for family in ["prompt_qwen", "classic", "qwen_vis"]:
        family_df = df[(df["family"] == family) & (df["stage"] == "after")]
        if family_df.empty:
            continue
        best = family_df.sort_values("macro_f1_mean", ascending=False).iloc[0].to_dict()
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows)
    if best_df.empty:
        return
    best_df["label"] = best_df.apply(canonical_label, axis=1)
    order = ["Prompt-Qwen Transfer", "Classic Transfer Best", "QwenVis Causal"]
    best_df["label"] = pd.Categorical(best_df["label"], categories=order, ordered=True)
    best_df = best_df.sort_values("label")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    palette = {
        "Prompt-Qwen Transfer": COLORS["prompt"],
        "Classic Transfer Best": COLORS["classic"],
        "QwenVis Causal": COLORS["qwen_causal"],
    }
    for ax, metric, title in [
        (axes[0], "macro_f1_mean", "Macro-F1"),
        (axes[1], "balanced_accuracy_mean", "Balanced Accuracy"),
    ]:
        sns.barplot(data=best_df, x="label", y=metric, hue="label", dodge=False, palette=palette, legend=False, ax=ax)
        add_errorbars(ax, best_df, metric, metric.replace("_mean", "_std"))
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Data2 External Transfer: Best Model by Family", fontsize=18, fontweight="bold")
    save_figure(fig, "fig05_data2_transfer_best")


def fig_per_class_heatmap(per_class: pd.DataFrame) -> None:
    df = per_class.copy()
    keep = ["Prompt-Qwen", "Classic Best", "QwenVis Baseline", "QwenVis Causal"]
    df = df[df["display_name"].isin(keep)].copy()
    df = df.groupby(["display_name", "class_name"], as_index=False)["recall"].mean()
    pivot = df.pivot(index="display_name", columns="class_name", values="recall").reindex(keep)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=1, cbar_kws={"label": "Recall"}, ax=ax)
    ax.set_title("Per-Class Recall on Data1 Official-CV Test", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "fig06_per_class_recall_heatmap")


def fig_data_protocol() -> None:
    payload = json.loads((CACHE_DIR / "paper_protocol_payload.json").read_text(encoding="utf-8"))
    data1 = payload["data2_analysis"]["source"]
    data2 = payload["data2_analysis"]["target"]

    label_rows = []
    for domain_name, domain_payload in [("data1", data1), ("data2", data2)]:
        for label, count in domain_payload["label_counts"].items():
            label_rows.append({"domain": domain_name, "label": label, "count": count})
    role_rows = [{"role": role, "count": count} for role, count in data2["role_counts"].items()]
    labels_df = pd.DataFrame(label_rows)
    roles_df = pd.DataFrame(role_rows).sort_values("count", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.barplot(data=labels_df, x="label", y="count", hue="domain", palette=["#457B9D", "#E76F51"], ax=axes[0])
    axes[0].set_title("Label Distribution: Data1 vs Data2")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Samples")

    sns.barplot(data=roles_df, x="count", y="role", color="#6C9A8B", ax=axes[1])
    axes[1].set_title("Data2 Role Distribution")
    axes[1].set_xlabel("Samples")
    axes[1].set_ylabel("")

    fig.suptitle("Protocol and Data Distribution Overview", fontsize=18, fontweight="bold")
    save_figure(fig, "fig07_protocol_distribution")


def export_paper_tables(metrics: pd.DataFrame) -> None:
    data1_main = metrics[
        (metrics["benchmark"] == "data1_official_cv")
        & (metrics["task_mode"] == "3class")
        & (metrics["split"] == "test")
        & (metrics["stage"] == "after")
    ].copy()
    data1_main["paper_label"] = data1_main.apply(canonical_label, axis=1)
    data1_main.to_csv(EXPORT_DIR / "table_data1_main_results.csv", index=False)

    transfer_main = metrics[
        (metrics["benchmark"] == "data2_transfer")
        & (metrics["task_mode"] == "3class")
        & (metrics["split"] == "transfer_test")
        & (metrics["stage"] == "after")
    ].copy()
    transfer_main["paper_label"] = transfer_main.apply(canonical_label, axis=1)
    transfer_main.to_csv(EXPORT_DIR / "table_data2_transfer_results.csv", index=False)


def main() -> None:
    set_style()
    ensure_dirs()
    inputs = load_inputs()
    metrics = inputs["metrics"]
    fig_data1_main_results(inputs["data1_focus"])
    fig_qwen_input_ablation(inputs["qwen_ablation"])
    fig_causal_gain(metrics)
    fig_last2_before_after(metrics)
    fig_transfer_family_best(inputs["transfer_focus"])
    fig_per_class_heatmap(inputs["per_class"])
    fig_data_protocol()
    export_paper_tables(metrics)
    summary = {
        "figures": sorted([p.name for p in EXPORT_DIR.glob("fig*.png")]),
        "tables": sorted([p.name for p in EXPORT_DIR.glob("table_*.csv")]),
    }
    (EXPORT_DIR / "plot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
