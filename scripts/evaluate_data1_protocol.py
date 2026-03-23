#!/usr/bin/env python3
"""Evaluate the data1 grouped protocol and generate bilingual comparison reports."""

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STAGES = ["before", "after"]
MODES = ["2class", "3class"]
SPLITS = ["val", "test"]
REPORT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "unknown_rate",
]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def normalize_model_arg(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def resolve_requested(arg_value: str, all_values: List[str]) -> List[str]:
    return list(all_values) if arg_value in {"all", "both"} else [arg_value]


def build_experiment_specs(
    args: argparse.Namespace,
    requested_modes: List[str],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    if "2class" in requested_modes and args.run_root_2class:
        specs.append(
            {
                "experiment": "2class_baseline",
                "experiment_label": "2Class Baseline / 二分类 Baseline",
                "mode": "2class",
                "method": "baseline",
                "run_root": Path(args.run_root_2class).resolve(),
            }
        )

    baseline_3class_root = args.run_root_3class_baseline or args.run_root_3class
    if "3class" in requested_modes and baseline_3class_root:
        specs.append(
            {
                "experiment": "3class_baseline",
                "experiment_label": "3Class Baseline / 三分类 Baseline",
                "mode": "3class",
                "method": "baseline",
                "run_root": Path(baseline_3class_root).resolve(),
            }
        )

    if "3class" in requested_modes and args.run_root_3class_causal:
        specs.append(
            {
                "experiment": "3class_causal",
                "experiment_label": "3Class Causal / 三分类 Causal",
                "mode": "3class",
                "method": "causal",
                "run_root": Path(args.run_root_3class_causal).resolve(),
            }
        )

    return specs


def discover_folds(protocol_root: Path, scope: str, fold_arg: str) -> List[str]:
    if scope == "quick_dev":
        return ["quick_dev"]

    if fold_arg != "all":
        return [fold_arg]

    cv_root = protocol_root / "official_cv"
    folds = sorted(path.name for path in cv_root.glob("fold_*") if path.is_dir())
    if not folds:
        raise FileNotFoundError(f"未在 {cv_root} 下发现任何 fold_* 目录")
    return folds


def manifest_path_for(protocol_root: Path, scope: str, fold_name: str, split: str) -> Path:
    if scope == "quick_dev":
        manifest_path = protocol_root / "quick_dev" / f"{split}_manifest.jsonl"
    else:
        manifest_path = protocol_root / "official_cv" / fold_name / f"{split}_manifest.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    return manifest_path


def fold_run_dir(run_root: Path, scope: str, fold_name: str) -> Path:
    return run_root if scope == "quick_dev" else run_root / fold_name


def combo_output_dir(base_run_dir: Path, stage: str, mode: str, split: str) -> Path:
    return base_run_dir / "evals_protocol" / stage / mode / split


def load_metrics(output_dir: Path) -> Dict[str, Any]:
    metrics_path = output_dir / "metrics_summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"缺少评估结果: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    folded_path = output_dir / "folded_2class" / "metrics_summary.json"
    if "folded_2class" not in metrics and folded_path.exists():
        with open(folded_path, "r", encoding="utf-8") as f:
            metrics["folded_2class"] = json.load(f)

    return metrics


def run_single_eval(
    args: argparse.Namespace,
    repo_root: Path,
    experiment_spec: Dict[str, Any],
    scope: str,
    fold_name: str,
    stage: str,
    split: str,
) -> Dict[str, Any]:
    mode = experiment_spec["mode"]
    run_root = experiment_spec["run_root"]
    base_run_dir = fold_run_dir(run_root, scope, fold_name)
    base_run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_path_for(Path(args.protocol_root).resolve(), scope, fold_name, split)
    output_dir = combo_output_dir(base_run_dir, stage, mode, split)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics_summary.json"
    if args.skip_existing and metrics_path.exists():
        print(
            f"[skip] {experiment_spec['experiment']}/{scope}/{fold_name}/{stage}/{mode}/{split} -> {metrics_path}"
        )
        return load_metrics(output_dir)

    cmd = [
        sys.executable,
        "src/evaluate_qwen2vl_slm.py",
        "--model_name_or_path",
        normalize_model_arg(args.model),
        "--manifest_path",
        str(manifest_path),
        "--output_dir",
        str(output_dir),
        "--eval_name",
        f"{scope}_{fold_name}_{split}",
        "--target_mode",
        mode,
        "--token_style",
        args.token_style,
        "--batch_size",
        str(args.batch_size),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--compute_candidate_scores",
        str(args.compute_candidate_scores).lower(),
        "--prediction_mode",
        args.prediction_mode,
        "--prompt_alignment",
        args.prompt_alignment,
        "--load_in_4bit",
        str(args.load_in_4bit).lower(),
        "--load_in_8bit",
        str(args.load_in_8bit).lower(),
        "--bf16",
        str(args.bf16).lower(),
        "--fp16",
        str(args.fp16).lower(),
    ]

    if args.cache_dir:
        cmd.extend(["--cache_dir", str(Path(args.cache_dir).expanduser())])
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    if stage == "after":
        adapter_path = base_run_dir / "adapter_best"
        if not adapter_path.exists():
            raise FileNotFoundError(f"after 评估需要 adapter_best，但未找到: {adapter_path}")
        cmd.extend(["--adapter_path", str(adapter_path)])

    print("[run]", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)
    return load_metrics(output_dir)


def safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def format_metric(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def format_mean_std(values: List[float], digits: int = 4) -> str:
    if not values:
        return "-"
    return f"{safe_mean(values):.{digits}f} ± {safe_std(values):.{digits}f}"


def aggregate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (
            record["experiment"],
            record["mode"],
            record["stage"],
            record["split"],
        )
        grouped.setdefault(key, []).append(record)

    aggregates: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        metrics_lists = {
            metric_name: [float(item["metrics"].get(metric_name, 0.0)) for item in items]
            for metric_name in REPORT_METRICS
        }
        aggregate = {
            "experiment": key[0],
            "experiment_label": items[0]["experiment_label"],
            "mode": key[1],
            "stage": key[2],
            "split": key[3],
            "fold_count": len(items),
            "n_samples_total": sum(int(item["metrics"].get("n_samples", 0)) for item in items),
            "n_samples_mean": safe_mean([int(item["metrics"].get("n_samples", 0)) for item in items]),
            "metrics": {
                metric_name: {
                    "mean": safe_mean(values),
                    "std": safe_std(values),
                }
                for metric_name, values in metrics_lists.items()
            },
        }

        folded_items = [
            item["metrics"]["folded_2class"]
            for item in items
            if item["metrics"].get("folded_2class")
        ]
        if folded_items:
            aggregate["folded_2class"] = {
                metric_name: {
                    "mean": safe_mean([float(metrics.get(metric_name, 0.0)) for metrics in folded_items]),
                    "std": safe_std([float(metrics.get(metric_name, 0.0)) for metrics in folded_items]),
                }
                for metric_name in REPORT_METRICS
            }

        aggregates.append(aggregate)

    return aggregates


def aggregate_index(aggregates: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    return {
        (item["experiment"], item["stage"], item["split"]): item
        for item in aggregates
    }


def write_report_files(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    aggregates: List[Dict[str, Any]],
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = report_dir / "comparison_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "records": records,
                "aggregates": aggregates,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_csv_path = report_dir / "comparison_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "scope",
                "fold",
                "stage",
                "mode",
                "split",
                "n_samples",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "unknown_rate",
                "output_dir",
            ]
        )
        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                [
                    record["experiment"],
                    record["scope"],
                    record["fold"],
                    record["stage"],
                    record["mode"],
                    record["split"],
                    metrics.get("n_samples", 0),
                    metrics.get("accuracy", 0.0),
                    metrics.get("balanced_accuracy", 0.0),
                    metrics.get("macro_f1", 0.0),
                    metrics.get("weighted_f1", 0.0),
                    metrics.get("unknown_rate", 0.0),
                    record["output_dir"],
                ]
            )

    agg_index = aggregate_index(aggregates)
    report_md_path = report_dir / "comparison_report.zh_en.md"
    legacy_report_md_path = report_dir / "comparison_report.md"
    lines: List[str] = []
    lines.append("# Data1 Protocol Evaluation Report / Data1 协议评估报告\n\n")
    lines.append(f"Generated / 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append(f"Scope / 协议范围: `{args.scope}`\n\n")
    lines.append(f"Protocol Root / 协议目录: `{Path(args.protocol_root).resolve()}`\n\n")
    lines.append(f"Model / 基础模型: `{normalize_model_arg(args.model)}`\n\n")
    lines.append(f"Prompt Alignment / Prompt 对齐方式: `{args.prompt_alignment}`\n\n")
    if args.run_root_2class:
        lines.append(f"2Class Run Root / 二分类运行目录: `{Path(args.run_root_2class).resolve()}`\n\n")
    baseline_3class_root = args.run_root_3class_baseline or args.run_root_3class
    if baseline_3class_root:
        lines.append(f"3Class Baseline Run Root / 三分类 Baseline 运行目录: `{Path(baseline_3class_root).resolve()}`\n\n")
    if args.run_root_3class_causal:
        lines.append(f"3Class Causal Run Root / 三分类 Causal 运行目录: `{Path(args.run_root_3class_causal).resolve()}`\n\n")

    lines.append("## Metric Guide / 指标说明\n\n")
    lines.append("- `accuracy`: overall accuracy / 总体准确率。\n")
    lines.append("- `balanced_accuracy`: mean recall across classes / 各类别召回率平均，更适合不均衡分类。\n")
    lines.append("- `macro_f1`: unweighted mean F1 across classes / 各类别 F1 的简单平均，推荐作为主指标。\n")
    lines.append("- `weighted_f1`: support-weighted F1 / 按类别样本数加权的 F1。\n")
    lines.append("- `unknown_rate`: unparsable prediction rate / 无法解析为合法标签的预测比例。\n")
    lines.append("- `folded_2class`: fold `HEW/LEL` into `abnormal` / 将 `HEW/LEL` 折叠为 `abnormal` 的辅助异常识别指标。\n\n")
    lines.append("- `prompt_alignment=train_validation`: use the same prompt-prefix truncation as train-time validation / 与训练验证使用同一条 prompt 前缀截断逻辑。\n\n")

    lines.append("## Per-Fold Coverage / 各折覆盖结果\n\n")
    lines.append("| Experiment | Scope | Fold | Stage | Mode | Split | Samples | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Unknown Rate |\n")
    lines.append("|------------|-------|------|-------|------|-------|---------|----------|--------------|----------|-------------|--------------|\n")
    for record in records:
        metrics = record["metrics"]
        lines.append(
            "| {experiment} | {scope} | {fold} | {stage} | {mode} | {split} | {n_samples} | {accuracy} | {balanced_accuracy} | {macro_f1} | {weighted_f1} | {unknown_rate} |\n".format(
                experiment=record["experiment"],
                scope=record["scope"],
                fold=record["fold"],
                stage=record["stage"],
                mode=record["mode"],
                split=record["split"],
                n_samples=metrics.get("n_samples", 0),
                accuracy=format_metric(metrics.get("accuracy")),
                balanced_accuracy=format_metric(metrics.get("balanced_accuracy")),
                macro_f1=format_metric(metrics.get("macro_f1")),
                weighted_f1=format_metric(metrics.get("weighted_f1")),
                unknown_rate=format_metric(metrics.get("unknown_rate")),
            )
        )
    lines.append("\n")

    lines.append("## Aggregate Mean ± Std / 汇总均值 ± 标准差\n\n")
    lines.append("| Experiment | Stage | Mode | Split | Folds | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Unknown Rate |\n")
    lines.append("|------------|-------|------|-------|-------|----------|--------------|----------|-------------|--------------|\n")
    for aggregate in aggregates:
        lines.append(
            "| {experiment} | {stage} | {mode} | {split} | {fold_count} | {accuracy} | {balanced_accuracy} | {macro_f1} | {weighted_f1} | {unknown_rate} |\n".format(
                experiment=aggregate["experiment"],
                stage=aggregate["stage"],
                mode=aggregate["mode"],
                split=aggregate["split"],
                fold_count=aggregate["fold_count"],
                accuracy=f"{aggregate['metrics']['accuracy']['mean']:.4f} ± {aggregate['metrics']['accuracy']['std']:.4f}",
                balanced_accuracy=f"{aggregate['metrics']['balanced_accuracy']['mean']:.4f} ± {aggregate['metrics']['balanced_accuracy']['std']:.4f}",
                macro_f1=f"{aggregate['metrics']['macro_f1']['mean']:.4f} ± {aggregate['metrics']['macro_f1']['std']:.4f}",
                weighted_f1=f"{aggregate['metrics']['weighted_f1']['mean']:.4f} ± {aggregate['metrics']['weighted_f1']['std']:.4f}",
                unknown_rate=f"{aggregate['metrics']['unknown_rate']['mean']:.4f} ± {aggregate['metrics']['unknown_rate']['std']:.4f}",
            )
        )
    lines.append("\n")

    experiment_labels = {
        aggregate["experiment"]: aggregate["experiment_label"]
        for aggregate in aggregates
    }
    experiment_order = ["2class_baseline", "3class_baseline", "3class_causal"]

    before_after_pairs = []
    for experiment in experiment_order:
        for split in SPLITS:
            before_metrics = agg_index.get((experiment, "before", split))
            after_metrics = agg_index.get((experiment, "after", split))
            if before_metrics and after_metrics:
                before_after_pairs.append((experiment, split, before_metrics, after_metrics))

    if before_after_pairs:
        lines.append("## Before vs After / 微调前后对比\n\n")
        for experiment, split, before_metrics, after_metrics in before_after_pairs:
            lines.append(f"### {experiment_labels.get(experiment, experiment)} / {split}\n\n")
            lines.append("| Metric | Before Mean | After Mean | Delta |\n")
            lines.append("|--------|-------------|------------|-------|\n")
            for metric_name in REPORT_METRICS:
                before_value = before_metrics["metrics"][metric_name]["mean"]
                after_value = after_metrics["metrics"][metric_name]["mean"]
                delta = after_value - before_value
                lines.append(
                    f"| {metric_name} | {before_value:.4f} | {after_value:.4f} | {delta:.4f} |\n"
                )
            lines.append("\n")

            if before_metrics.get("folded_2class") and after_metrics.get("folded_2class"):
                lines.append("Folded 2Class / 三分类折叠成二分类\n\n")
                lines.append("| Metric | Before Mean | After Mean | Delta |\n")
                lines.append("|--------|-------------|------------|-------|\n")
                for metric_name in REPORT_METRICS:
                    before_value = before_metrics["folded_2class"][metric_name]["mean"]
                    after_value = after_metrics["folded_2class"][metric_name]["mean"]
                    delta = after_value - before_value
                    lines.append(
                        f"| folded::{metric_name} | {before_value:.4f} | {after_value:.4f} | {delta:.4f} |\n"
                    )
                lines.append("\n")

    mode_pairs = []
    for stage in STAGES:
        for split in SPLITS:
            metrics_2 = agg_index.get(("2class_baseline", stage, split))
            for experiment in ["3class_baseline", "3class_causal"]:
                metrics_3 = agg_index.get((experiment, stage, split))
                if metrics_2 and metrics_3 and metrics_3.get("folded_2class"):
                    mode_pairs.append((stage, split, experiment, metrics_2, metrics_3))

    if mode_pairs:
        lines.append("## 2Class Baseline vs 3Class Variants / 二分类 Baseline 与三分类变体对比\n\n")
        for stage, split, experiment, metrics_2, metrics_3 in mode_pairs:
            lines.append(f"### {stage} / {split} / {experiment_labels.get(experiment, experiment)}\n\n")
            lines.append("| Metric | 2Class Baseline Mean | 3Class Main Mean | 3Class Folded Mean | Delta(2Class-Folded) |\n")
            lines.append("|--------|-------------|------------------|--------------------|----------------------|\n")
            for metric_name in REPORT_METRICS:
                main_2 = metrics_2["metrics"][metric_name]["mean"]
                main_3 = metrics_3["metrics"][metric_name]["mean"]
                folded_3 = metrics_3["folded_2class"][metric_name]["mean"]
                lines.append(
                    f"| {metric_name} | {main_2:.4f} | {main_3:.4f} | {folded_3:.4f} | {main_2 - folded_3:.4f} |\n"
                )
            lines.append("\n")

    causal_pairs = []
    for stage in STAGES:
        for split in SPLITS:
            baseline_metrics = agg_index.get(("3class_baseline", stage, split))
            causal_metrics = agg_index.get(("3class_causal", stage, split))
            if baseline_metrics and causal_metrics:
                causal_pairs.append((stage, split, baseline_metrics, causal_metrics))

    if causal_pairs:
        lines.append("## 3Class Baseline vs 3Class Causal / 三分类 Baseline 与 Causal 对比\n\n")
        for stage, split, baseline_metrics, causal_metrics in causal_pairs:
            lines.append(f"### {stage} / {split}\n\n")
            lines.append("| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |\n")
            lines.append("|--------|---------------|-------------|------------------------|\n")
            for metric_name in REPORT_METRICS:
                baseline_value = baseline_metrics["metrics"][metric_name]["mean"]
                causal_value = causal_metrics["metrics"][metric_name]["mean"]
                lines.append(
                    f"| {metric_name} | {baseline_value:.4f} | {causal_value:.4f} | {causal_value - baseline_value:.4f} |\n"
                )
            lines.append("\n")

            if baseline_metrics.get("folded_2class") and causal_metrics.get("folded_2class"):
                lines.append("Folded 2Class / 三分类折叠成二分类\n\n")
                lines.append("| Metric | Baseline Mean | Causal Mean | Delta(Causal-Baseline) |\n")
                lines.append("|--------|---------------|-------------|------------------------|\n")
                for metric_name in REPORT_METRICS:
                    baseline_value = baseline_metrics["folded_2class"][metric_name]["mean"]
                    causal_value = causal_metrics["folded_2class"][metric_name]["mean"]
                    lines.append(
                        f"| folded::{metric_name} | {baseline_value:.4f} | {causal_value:.4f} | {causal_value - baseline_value:.4f} |\n"
                    )
                lines.append("\n")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    with open(legacy_report_md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 data1 grouped protocol，并生成中英双语聚合报告"
    )
    parser.add_argument("--scope", choices=["quick_dev", "official_cv"], default="official_cv")
    parser.add_argument("--stage", choices=["before", "after", "both"], default="both")
    parser.add_argument("--mode", choices=["2class", "3class", "all"], default="all")
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--fold", default="all", help="quick_dev 下忽略；official_cv 可设 fold_01 或 all")
    parser.add_argument(
        "--protocol_root",
        default="/home/gjw/code/SLM_data/processed_qwen_data1_protocol",
    )
    parser.add_argument("--run_root_2class", default=None, help="二分类 baseline 训练根目录")
    parser.add_argument("--run_root_3class", default=None, help="兼容旧参数：等价于 --run_root_3class_baseline")
    parser.add_argument("--run_root_3class_baseline", default=None, help="三分类 baseline 训练根目录")
    parser.add_argument("--run_root_3class_causal", default=None, help="三分类 causal 训练根目录")
    parser.add_argument(
        "--report_dir",
        default=None,
        help="报告输出目录，默认写到 ./runs/data1_protocol_reports/<scope>",
    )
    parser.add_argument(
        "--model",
        default="/home/gjw/code/model_cache/Qwen/Qwen2-VL-2B-Instruct",
    )
    parser.add_argument("--token_style", default="letters", choices=["letters", "words"])
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--compute_candidate_scores", type=str2bool, default=True)
    parser.add_argument(
        "--prediction_mode",
        choices=["auto", "generate", "candidate_score"],
        default="candidate_score",
    )
    parser.add_argument(
        "--prompt_alignment",
        choices=["train_validation", "inference"],
        default="train_validation",
        help="正式评估的 prompt 对齐方式；默认与训练验证一致",
    )
    parser.add_argument("--load_in_4bit", type=str2bool, default=False)
    parser.add_argument("--load_in_8bit", type=str2bool, default=False)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--skip_existing", type=str2bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    protocol_root = Path(args.protocol_root).resolve()

    requested_modes = resolve_requested(args.mode, MODES)
    requested_stages = resolve_requested(args.stage, STAGES)
    requested_splits = resolve_requested(args.split, SPLITS)
    folds = discover_folds(protocol_root, args.scope, args.fold)
    experiment_specs = build_experiment_specs(args, requested_modes)

    if "2class" in requested_modes and not any(spec["mode"] == "2class" for spec in experiment_specs):
        raise ValueError("mode 包含 2class，但未提供 --run_root_2class")
    if "3class" in requested_modes and not any(spec["mode"] == "3class" for spec in experiment_specs):
        raise ValueError("mode 包含 3class，但未提供 --run_root_3class_baseline 或 --run_root_3class_causal")

    records: List[Dict[str, Any]] = []
    for experiment_spec in experiment_specs:
        for fold_name in folds:
            for stage in requested_stages:
                for split in requested_splits:
                    metrics = run_single_eval(
                        args=args,
                        repo_root=repo_root,
                        experiment_spec=experiment_spec,
                        scope=args.scope,
                        fold_name=fold_name,
                        stage=stage,
                        split=split,
                    )
                    records.append(
                        {
                            "experiment": experiment_spec["experiment"],
                            "experiment_label": experiment_spec["experiment_label"],
                            "scope": args.scope,
                            "fold": fold_name,
                            "stage": stage,
                            "mode": experiment_spec["mode"],
                            "method": experiment_spec["method"],
                            "split": split,
                            "output_dir": str(
                                combo_output_dir(
                                    fold_run_dir(experiment_spec["run_root"], args.scope, fold_name),
                                    stage,
                                    experiment_spec["mode"],
                                    split,
                                )
                            ),
                            "metrics": metrics,
                        }
                    )

    records.sort(
        key=lambda item: (
            item["experiment"],
            item["fold"],
            item["stage"],
            item["split"],
        )
    )
    aggregates = aggregate_records(records)

    if args.report_dir:
        report_dir = Path(args.report_dir).resolve()
    else:
        report_dir = (repo_root / "runs" / "data1_protocol_reports" / args.scope).resolve()
    write_report_files(args=args, records=records, aggregates=aggregates, report_dir=report_dir)

    print("=" * 72)
    print("Data1 protocol evaluation finished / Data1 协议评估完成")
    print(f"Report dir / 报告目录: {report_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
