#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.path_utils import resolve_path

OUTPUT_ROOT = REPO_ROOT / "paper_visualization"
CACHE_DIR = OUTPUT_ROOT / "cache"
EXPORT_DIR = OUTPUT_ROOT / "exports"
RUNS_ROOT = resolve_path(os.environ.get("PROJECT_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))
DATA2_ANALYSIS_ROOT = resolve_path(os.environ.get("SLM_DATA2_TRANSFER_ROOT", "${PROJECT_ROOT}/data_placeholder/data2_transfer"))

DATA1_PROTOCOL_SUMMARY = RUNS_ROOT / "data1_protocol_reports/official_cv/comparison_summary.json"
CLASSIC_SUMMARY = RUNS_ROOT / "classic_mm_baselines/reports/official_cv_extended/comparison_summary.json"
QWEN_VIS_SUMMARY = RUNS_ROOT / "qwen_vis_fusion/reports/official_cv/comparison_summary.json"
QWEN_VIS_LAST2_SUMMARY = RUNS_ROOT / "qwen_vis_fusion/reports/official_cv_last2/comparison_summary.json"
DATA2_TRANSFER_SUMMARY = RUNS_ROOT / "data2_transfer_eval/reports/comparison_summary.json"

DATA1_SPLIT_DESCRIPTION = RUNS_ROOT / "data1_protocol_reports/official_cv/split_description.json"
DATA2_ANALYSIS = DATA2_ANALYSIS_ROOT / "data2_analysis.json"

QWEN_VIS_ABLATION_RUNS = [
    RUNS_ROOT / "qwen_vis_fusion/official_cv_3class_rgb_view1_baseline",
    RUNS_ROOT / "qwen_vis_fusion/official_cv_3class_rgb_dual_baseline",
    RUNS_ROOT / "qwen_vis_fusion/official_cv_3class_ir_only_baseline",
    RUNS_ROOT / "qwen_vis_fusion/official_cv_3class_triple_view_baseline",
    RUNS_ROOT / "qwen_vis_fusion/official_cv_3class_rgb_dual_causal",
]


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_value(data: Dict[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def normalize_data1_protocol() -> List[Dict[str, Any]]:
    payload = load_json(DATA1_PROTOCOL_SUMMARY)
    rows: List[Dict[str, Any]] = []
    for item in payload["aggregates"]:
        rows.append(
            {
                "benchmark": "data1_official_cv",
                "family": "prompt_qwen",
                "experiment_name": item["experiment"],
                "display_name": item["experiment_label"],
                "task_mode": item["mode"],
                "method": "causal" if "causal" in item["experiment"] else "baseline",
                "input_mode": "triple_view",
                "stage": item["stage"],
                "split": item["split"],
                "folds": item["fold_count"],
                "accuracy_mean": safe_value(item, "metrics", "accuracy", "mean", default=0.0),
                "accuracy_std": safe_value(item, "metrics", "accuracy", "std", default=0.0),
                "balanced_accuracy_mean": safe_value(item, "metrics", "balanced_accuracy", "mean", default=0.0),
                "balanced_accuracy_std": safe_value(item, "metrics", "balanced_accuracy", "std", default=0.0),
                "macro_f1_mean": safe_value(item, "metrics", "macro_f1", "mean", default=0.0),
                "macro_f1_std": safe_value(item, "metrics", "macro_f1", "std", default=0.0),
                "weighted_f1_mean": safe_value(item, "metrics", "weighted_f1", "mean", default=0.0),
                "weighted_f1_std": safe_value(item, "metrics", "weighted_f1", "std", default=0.0),
                "folded_2class_macro_f1_mean": safe_value(item, "metrics", "folded_2class_macro_f1", "mean", default=0.0),
                "folded_2class_macro_f1_std": safe_value(item, "metrics", "folded_2class_macro_f1", "std", default=0.0),
                "source_path": str(DATA1_PROTOCOL_SUMMARY),
            }
        )
    return rows


def normalize_summary_list(path: Path, benchmark: str, family_override: Optional[str] = None) -> List[Dict[str, Any]]:
    payload = load_json(path)
    rows: List[Dict[str, Any]] = []
    for item in payload:
        family = family_override or item.get("family", "unknown")
        rows.append(
            {
                "benchmark": benchmark,
                "family": family,
                "experiment_name": item["experiment_name"],
                "display_name": item["experiment_name"],
                "task_mode": item["task_mode"],
                "method": item.get("method", "baseline"),
                "input_mode": item.get("input_mode", ""),
                "stage": item.get("stage", "after"),
                "split": item["split"],
                "folds": item["folds"],
                "accuracy_mean": item["accuracy_mean"],
                "accuracy_std": item["accuracy_std"],
                "balanced_accuracy_mean": item["balanced_accuracy_mean"],
                "balanced_accuracy_std": item["balanced_accuracy_std"],
                "macro_f1_mean": item["macro_f1_mean"],
                "macro_f1_std": item["macro_f1_std"],
                "weighted_f1_mean": item["weighted_f1_mean"],
                "weighted_f1_std": item["weighted_f1_std"],
                "folded_2class_macro_f1_mean": item.get("folded_2class_macro_f1_mean", 0.0),
                "folded_2class_macro_f1_std": item.get("folded_2class_macro_f1_std", 0.0),
                "unfreeze_last_n_blocks": item.get("unfreeze_last_n_blocks", 0),
                "source_path": str(path),
                "run_root": item.get("run_root", ""),
            }
        )
    return rows


def load_qwen_vis_run(run_root: Path) -> List[Dict[str, Any]]:
    payload = load_json(run_root / "run_summary.json")
    rows: List[Dict[str, Any]] = []
    for fold_result in payload["fold_results"]:
        metric_specs = [
            ("before_val_metrics", "before", "val"),
            ("before_test_metrics", "before", "test"),
            ("val_metrics", "after", "val"),
            ("test_metrics", "after", "test"),
        ]
        for metric_key, stage, split in metric_specs:
            metrics = fold_result.get(metric_key)
            if metrics is None:
                continue
            rows.append(
                {
                    "benchmark": "data1_official_cv",
                    "family": "qwen_vis",
                    "experiment_name": payload["experiment_name"],
                    "display_name": payload["experiment_name"],
                    "task_mode": payload["task_mode"],
                    "method": payload["method"],
                    "input_mode": payload["input_mode"],
                    "stage": stage,
                    "split": split,
                    "folds": 1,
                    "accuracy_mean": metrics["accuracy"],
                    "accuracy_std": 0.0,
                    "balanced_accuracy_mean": metrics["balanced_accuracy"],
                    "balanced_accuracy_std": 0.0,
                    "macro_f1_mean": metrics["macro_f1"],
                    "macro_f1_std": 0.0,
                    "weighted_f1_mean": metrics["weighted_f1"],
                    "weighted_f1_std": 0.0,
                    "folded_2class_macro_f1_mean": safe_value(metrics, "folded_2class", "macro_f1", default=0.0),
                    "folded_2class_macro_f1_std": 0.0,
                    "unfreeze_last_n_blocks": payload.get("unfreeze_last_n_blocks", 0),
                    "source_path": str(run_root / "run_summary.json"),
                    "run_root": str(run_root),
                    "fold": fold_result["fold"],
                }
            )
    return rows


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = [
        "benchmark",
        "family",
        "experiment_name",
        "display_name",
        "task_mode",
        "method",
        "input_mode",
        "stage",
        "split",
        "unfreeze_last_n_blocks",
        "run_root",
        "source_path",
    ]
    metric_cols = [
        "accuracy_mean",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "weighted_f1_mean",
        "folded_2class_macro_f1_mean",
    ]
    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    agg.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    result_rows: List[Dict[str, Any]] = []
    for _, row in agg.iterrows():
        result_rows.append(
            {
                "benchmark": row["benchmark"],
                "family": row["family"],
                "experiment_name": row["experiment_name"],
                "display_name": row["display_name"],
                "task_mode": row["task_mode"],
                "method": row["method"],
                "input_mode": row["input_mode"],
                "stage": row["stage"],
                "split": row["split"],
                "unfreeze_last_n_blocks": int(row["unfreeze_last_n_blocks"]) if pd.notna(row["unfreeze_last_n_blocks"]) else 0,
                "run_root": row["run_root"],
                "source_path": row["source_path"],
                "folds": int(row["accuracy_mean_count"]),
                "accuracy_mean": float(row["accuracy_mean_mean"]),
                "accuracy_std": float(0.0 if pd.isna(row["accuracy_mean_std"]) else row["accuracy_mean_std"]),
                "balanced_accuracy_mean": float(row["balanced_accuracy_mean_mean"]),
                "balanced_accuracy_std": float(0.0 if pd.isna(row["balanced_accuracy_mean_std"]) else row["balanced_accuracy_mean_std"]),
                "macro_f1_mean": float(row["macro_f1_mean_mean"]),
                "macro_f1_std": float(0.0 if pd.isna(row["macro_f1_mean_std"]) else row["macro_f1_mean_std"]),
                "weighted_f1_mean": float(row["weighted_f1_mean_mean"]),
                "weighted_f1_std": float(0.0 if pd.isna(row["weighted_f1_mean_std"]) else row["weighted_f1_mean_std"]),
                "folded_2class_macro_f1_mean": float(row["folded_2class_macro_f1_mean_mean"]),
                "folded_2class_macro_f1_std": float(0.0 if pd.isna(row["folded_2class_macro_f1_mean_std"]) else row["folded_2class_macro_f1_mean_std"]),
            }
        )
    return result_rows


def collect_qwen_vis_ablation() -> List[Dict[str, Any]]:
    detail_rows: List[Dict[str, Any]] = []
    for run_root in QWEN_VIS_ABLATION_RUNS:
        summary_path = run_root / "run_summary.json"
        if summary_path.exists():
            detail_rows.extend(load_qwen_vis_run(run_root))
    return aggregate_rows(detail_rows)


def metrics_path_for_descriptor(descriptor: Dict[str, Any], fold: str) -> Optional[Path]:
    run_root = Path(descriptor["run_root"])
    if descriptor["kind"] == "prompt":
        return run_root / fold / "evals_protocol" / descriptor["stage"] / descriptor["task_mode"] / descriptor["split"] / "metrics_summary.json"
    if descriptor["kind"] == "classic":
        return run_root / fold / "evals" / descriptor["split"] / "metrics_summary.json"
    if descriptor["kind"] == "qwen_vis":
        stage = descriptor.get("stage", "after")
        staged = run_root / fold / "evals" / stage / descriptor["split"] / "metrics_summary.json"
        plain = run_root / fold / "evals" / descriptor["split"] / "metrics_summary.json"
        return staged if staged.exists() else plain
    return None


def predictions_path_for_descriptor(descriptor: Dict[str, Any], fold: str) -> Optional[Path]:
    run_root = Path(descriptor["run_root"])
    if descriptor["kind"] == "prompt":
        return run_root / fold / "evals_protocol" / descriptor["stage"] / descriptor["task_mode"] / descriptor["split"] / "predictions.jsonl"
    if descriptor["kind"] == "classic":
        return run_root / fold / "evals" / descriptor["split"] / "predictions.jsonl"
    if descriptor["kind"] == "qwen_vis":
        stage = descriptor.get("stage", "after")
        staged = run_root / fold / "evals" / stage / descriptor["split"] / "predictions.csv"
        plain = run_root / fold / "evals" / descriptor["split"] / "predictions.csv"
        return staged if staged.exists() else plain
    return None


def collect_per_class_and_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptors = [
        {
            "benchmark": "data1_official_cv",
            "family": "prompt_qwen",
            "display_name": "Prompt-Qwen",
            "experiment_name": "data1_official_cv_3class_baseline",
            "run_root": str(REPO_ROOT / "runs/data1_official_cv_3class_baseline"),
            "kind": "prompt",
            "task_mode": "3class",
            "stage": "after",
            "split": "test",
        },
        {
            "benchmark": "data1_official_cv",
            "family": "classic",
            "display_name": "Classic Best",
            "experiment_name": "resnet18_rgb_dual_mlp_concat_3class",
            "run_root": str(REPO_ROOT / "runs/classic_mm_baselines/official_cv_3class_rgb_dual_mlp_concat"),
            "kind": "classic",
            "task_mode": "3class",
            "stage": "after",
            "split": "test",
        },
        {
            "benchmark": "data1_official_cv",
            "family": "qwen_vis",
            "display_name": "QwenVis Baseline",
            "experiment_name": "official_cv_3class_rgb_dual_baseline",
            "run_root": str(REPO_ROOT / "runs/qwen_vis_fusion/official_cv_3class_rgb_dual_baseline"),
            "kind": "qwen_vis",
            "task_mode": "3class",
            "stage": "after",
            "split": "test",
        },
        {
            "benchmark": "data1_official_cv",
            "family": "qwen_vis",
            "display_name": "QwenVis Causal",
            "experiment_name": "official_cv_3class_rgb_dual_causal",
            "run_root": str(REPO_ROOT / "runs/qwen_vis_fusion/official_cv_3class_rgb_dual_causal"),
            "kind": "qwen_vis",
            "task_mode": "3class",
            "stage": "after",
            "split": "test",
        },
        {
            "benchmark": "data1_official_cv",
            "family": "qwen_vis",
            "display_name": "QwenVis Last2 Before",
            "experiment_name": "official_cv_3class_rgb_dual_last2blocks_causal",
            "run_root": str(REPO_ROOT / "runs/qwen_vis_fusion/official_cv_3class_rgb_dual_last2blocks_causal"),
            "kind": "qwen_vis",
            "task_mode": "3class",
            "stage": "before",
            "split": "test",
        },
        {
            "benchmark": "data1_official_cv",
            "family": "qwen_vis",
            "display_name": "QwenVis Last2 After",
            "experiment_name": "official_cv_3class_rgb_dual_last2blocks_causal",
            "run_root": str(REPO_ROOT / "runs/qwen_vis_fusion/official_cv_3class_rgb_dual_last2blocks_causal"),
            "kind": "qwen_vis",
            "task_mode": "3class",
            "stage": "after",
            "split": "test",
        },
        {
            "benchmark": "data1_official_cv",
            "family": "qwen_vis",
            "display_name": "QwenVis RGBv1",
            "experiment_name": "official_cv_3class_rgb_view1_baseline",
            "run_root": str(REPO_ROOT / "runs/qwen_vis_fusion/official_cv_3class_rgb_view1_baseline"),
            "kind": "qwen_vis",
            "task_mode": "3class",
            "stage": "after",
            "split": "test",
        },
        {
            "benchmark": "data1_official_cv",
            "family": "qwen_vis",
            "display_name": "QwenVis IR-only",
            "experiment_name": "official_cv_3class_ir_only_baseline",
            "run_root": str(REPO_ROOT / "runs/qwen_vis_fusion/official_cv_3class_ir_only_baseline"),
            "kind": "qwen_vis",
            "task_mode": "3class",
            "stage": "after",
            "split": "test",
        },
        {
            "benchmark": "data1_official_cv",
            "family": "qwen_vis",
            "display_name": "QwenVis Triple-View",
            "experiment_name": "official_cv_3class_triple_view_baseline",
            "run_root": str(REPO_ROOT / "runs/qwen_vis_fusion/official_cv_3class_triple_view_baseline"),
            "kind": "qwen_vis",
            "task_mode": "3class",
            "stage": "after",
            "split": "test",
        },
    ]

    per_class_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    for descriptor in descriptors:
        for fold_idx in range(1, 4):
            fold = f"fold_{fold_idx:02d}"
            metrics_path = metrics_path_for_descriptor(descriptor, fold)
            if metrics_path is None or not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            per_class = metrics.get("per_class", {})
            for class_name, class_metrics in per_class.items():
                per_class_rows.append(
                    {
                        "benchmark": descriptor["benchmark"],
                        "family": descriptor["family"],
                        "display_name": descriptor["display_name"],
                        "experiment_name": descriptor["experiment_name"],
                        "task_mode": descriptor["task_mode"],
                        "stage": descriptor["stage"],
                        "split": descriptor["split"],
                        "fold": fold,
                        "class_name": class_name,
                        "precision": class_metrics.get("precision", 0.0),
                        "recall": class_metrics.get("recall", 0.0),
                        "f1": class_metrics.get("f1", 0.0),
                        "support": class_metrics.get("support", 0),
                    }
                )

            pred_path = predictions_path_for_descriptor(descriptor, fold)
            if pred_path is None or not pred_path.exists():
                continue
            if pred_path.suffix == ".jsonl":
                rows = load_jsonl(pred_path)
            else:
                with pred_path.open("r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            for row in rows:
                prediction_rows.append(
                    {
                        "benchmark": descriptor["benchmark"],
                        "family": descriptor["family"],
                        "display_name": descriptor["display_name"],
                        "experiment_name": descriptor["experiment_name"],
                        "task_mode": descriptor["task_mode"],
                        "stage": descriptor["stage"],
                        "split": descriptor["split"],
                        "fold": fold,
                        "sample_id": row.get("sample_id"),
                        "true_label": row.get("true_label"),
                        "pred_label": row.get("pred_label"),
                        "condition_uid": row.get("condition_uid"),
                        "role": row.get("role"),
                    }
                )
    return pd.DataFrame(per_class_rows), pd.DataFrame(prediction_rows)


def build_focus_tables(metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    data1_3class = metrics_df[
        (metrics_df["benchmark"] == "data1_official_cv")
        & (metrics_df["task_mode"] == "3class")
        & (metrics_df["split"] == "test")
        & (
            (
                (metrics_df["family"] == "prompt_qwen")
                & (metrics_df["experiment_name"] == "3class_baseline")
                & (metrics_df["stage"] == "after")
            )
            | (
                (metrics_df["family"] == "classic")
                & (metrics_df["experiment_name"] == "resnet18_rgb_dual_mlp_concat_3class")
            )
            | (
                (metrics_df["family"] == "qwen_vis")
                & metrics_df["experiment_name"].isin(
                    [
                        "qwen_vis_rgb_dual_baseline_3class",
                        "qwen_vis_rgb_dual_causal_3class",
                        "qwen_vis_rgb_dual_last2blocks_baseline_3class",
                        "qwen_vis_rgb_dual_last2blocks_causal_3class",
                    ]
                )
            )
        )
    ].copy()

    qwen_input_ablation = metrics_df[
        (metrics_df["benchmark"] == "data1_official_cv")
        & (metrics_df["family"] == "qwen_vis")
        & (metrics_df["task_mode"] == "3class")
        & (metrics_df["split"] == "test")
        & (metrics_df["stage"] == "after")
        & (
            metrics_df["experiment_name"].isin(
                [
                    "qwen_vis_rgb_view1_baseline_3class",
                    "qwen_vis_rgb_dual_baseline_3class",
                    "qwen_vis_ir_only_baseline_3class",
                    "qwen_vis_triple_view_baseline_3class",
                    "qwen_vis_rgb_dual_causal_3class",
                ]
            )
        )
    ].copy()

    transfer_best = metrics_df[
        (metrics_df["benchmark"] == "data2_transfer")
        & (metrics_df["task_mode"] == "3class")
        & (metrics_df["split"] == "transfer_test")
        & (metrics_df["stage"] == "after")
    ].copy()

    return {
        "focus_data1_3class": data1_3class,
        "qwen_input_ablation_3class": qwen_input_ablation,
        "focus_data2_transfer": transfer_best,
    }


def main() -> None:
    ensure_dirs()

    rows: List[Dict[str, Any]] = []
    rows.extend(normalize_data1_protocol())
    rows.extend(normalize_summary_list(CLASSIC_SUMMARY, benchmark="data1_official_cv", family_override="classic"))
    rows.extend(normalize_summary_list(QWEN_VIS_SUMMARY, benchmark="data1_official_cv", family_override="qwen_vis"))
    rows.extend(normalize_summary_list(QWEN_VIS_LAST2_SUMMARY, benchmark="data1_official_cv", family_override="qwen_vis"))
    rows.extend(normalize_summary_list(DATA2_TRANSFER_SUMMARY, benchmark="data2_transfer"))
    rows.extend(collect_qwen_vis_ablation())

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.drop_duplicates(
        subset=[
            "benchmark",
            "family",
            "experiment_name",
            "task_mode",
            "method",
            "input_mode",
            "stage",
            "split",
            "unfreeze_last_n_blocks",
        ],
        keep="last",
    )
    metrics_df.to_csv(CACHE_DIR / "paper_metrics_all.csv", index=False)

    focus_tables = build_focus_tables(metrics_df)
    for name, df in focus_tables.items():
        df.to_csv(CACHE_DIR / f"{name}.csv", index=False)

    per_class_df, predictions_df = collect_per_class_and_predictions()
    per_class_df.to_csv(CACHE_DIR / "paper_per_class_selected.csv", index=False)
    predictions_df.to_csv(CACHE_DIR / "paper_predictions_selected.csv", index=False)

    protocol_payload = {
        "data1_split_description": load_json(DATA1_SPLIT_DESCRIPTION),
        "data2_analysis": load_json(DATA2_ANALYSIS),
    }
    (CACHE_DIR / "paper_protocol_payload.json").write_text(
        json.dumps(protocol_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "metrics_rows": len(metrics_df),
        "focus_tables": {name: len(df) for name, df in focus_tables.items()},
        "per_class_rows": len(per_class_df),
        "prediction_rows": len(predictions_df),
    }
    (CACHE_DIR / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
