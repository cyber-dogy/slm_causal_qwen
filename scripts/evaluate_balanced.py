#!/usr/bin/env python3
"""统一评估编排脚本: 支持 before/after、2class/3class、source/transfer，并生成对比报告。"""

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


SUBSET_TO_MANIFEST = {
    "source_test": "test_manifest.jsonl",
    "transfer_test": "transfer_test_manifest.jsonl",
}
STAGES = ["before", "after"]
MODES = ["2class", "3class"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 balanced 协议评估，并生成训练前后/二三分类对比报告"
    )
    parser.add_argument("--stage", choices=["before", "after", "both"], default="both")
    parser.add_argument("--mode", choices=["2class", "3class", "all"], default="all")
    parser.add_argument(
        "--subset",
        choices=["source_test", "transfer_test", "all"],
        default="all",
        help="评估 source_test(test_manifest) 或 transfer_test(transfer_test_manifest)",
    )
    parser.add_argument("--run_dir", default="./runs/exp_balanced_3class")
    parser.add_argument(
        "--data_dir",
        default="/home/gjw/code/SLM_data/processed_qwen_balanced",
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
    parser.add_argument("--load_in_4bit", type=str2bool, default=False)
    parser.add_argument("--load_in_8bit", type=str2bool, default=False)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument(
        "--skip_existing",
        type=str2bool,
        default=False,
        help="若输出目录已有 metrics_summary.json，则跳过该组合的重新评估",
    )
    return parser.parse_args()


def resolve_requested(arg_value: str, all_values: List[str]) -> List[str]:
    return list(all_values) if arg_value in {"all", "both"} else [arg_value]


def ensure_manifest(data_dir: Path, subset: str) -> Path:
    manifest_path = data_dir / SUBSET_TO_MANIFEST[subset]
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    return manifest_path


def combo_output_dir(run_dir: Path, stage: str, mode: str, subset: str) -> Path:
    return run_dir / "evals" / stage / mode / subset


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
    stage: str,
    mode: str,
    subset: str,
    repo_root: Path,
) -> Dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    manifest_path = ensure_manifest(data_dir, subset)
    output_dir = combo_output_dir(run_dir, stage, mode, subset)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics_summary.json"
    if args.skip_existing and metrics_path.exists():
        print(f"[skip] {stage}/{mode}/{subset} -> {metrics_path}")
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
        subset,
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
        adapter_path = run_dir / "adapter_best"
        if not adapter_path.exists():
            raise FileNotFoundError(f"after 评估需要 adapter_best，但未找到: {adapter_path}")
        cmd.extend(["--adapter_path", str(adapter_path)])

    print("[run]", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)
    return load_metrics(output_dir)


def collect_available_results(run_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for stage in STAGES:
        for mode in MODES:
            for subset in SUBSET_TO_MANIFEST:
                output_dir = combo_output_dir(run_dir, stage, mode, subset)
                metrics_path = output_dir / "metrics_summary.json"
                if not metrics_path.exists():
                    continue
                records.append(
                    {
                        "stage": stage,
                        "mode": mode,
                        "subset": subset,
                        "output_dir": str(output_dir),
                        "metrics": load_metrics(output_dir),
                    }
                )
    records.sort(key=lambda item: (item["stage"], item["mode"], item["subset"]))
    return records


def format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_index(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    return {
        (record["stage"], record["mode"], record["subset"]): record
        for record in records
    }


def write_report_files(records: List[Dict[str, Any]], report_dir: Path, args: argparse.Namespace) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    index = build_index(records)

    summary_json_path = report_dir / "comparison_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    summary_csv_path = report_dir / "comparison_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage",
            "mode",
            "subset",
            "n_samples",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "unknown_rate",
            "output_dir",
        ])
        for record in records:
            metrics = record["metrics"]
            writer.writerow([
                record["stage"],
                record["mode"],
                record["subset"],
                metrics.get("n_samples", 0),
                metrics.get("accuracy", 0.0),
                metrics.get("balanced_accuracy", 0.0),
                metrics.get("macro_f1", 0.0),
                metrics.get("weighted_f1", 0.0),
                metrics.get("unknown_rate", 0.0),
                record["output_dir"],
            ])

    report_md_path = report_dir / "comparison_report.md"
    lines: List[str] = []
    lines.append("# Balanced Evaluation Comparison Report / 平衡协议评估对比报告\n\n")
    lines.append(f"Generated / 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append(f"Run dir / 运行目录: `{Path(args.run_dir).resolve()}`\n\n")
    lines.append(f"Data dir / 数据目录: `{Path(args.data_dir).resolve()}`\n\n")
    lines.append(f"Model / 基础模型: `{normalize_model_arg(args.model)}`\n\n")
    lines.append("## Metric Guide / 指标说明\n\n")
    lines.append("- `accuracy`: overall accuracy / 总体准确率。\n")
    lines.append("- `balanced_accuracy`: mean recall across classes / 各类别召回率平均。\n")
    lines.append("- `macro_f1`: unweighted mean F1 across classes / 各类别 F1 的简单平均。\n")
    lines.append("- `weighted_f1`: support-weighted F1 / 按类别样本数加权的 F1。\n")
    lines.append("- `unknown_rate`: unparsable prediction rate / 无法解析成合法标签的预测比例。\n")
    lines.append("- `folded_2class`: folded anomaly-vs-normal view / 将 `HEW/LEL` 折叠为异常识别结果。\n\n")

    lines.append("## Coverage / 覆盖结果\n\n")
    lines.append("| Stage | Mode | Subset | Samples | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Unknown Rate | Output |\n")
    lines.append("|-------|------|--------|---------|----------|--------------|----------|-------------|--------------|--------|\n")
    for record in records:
        metrics = record["metrics"]
        lines.append(
            "| {stage} | {mode} | {subset} | {n_samples} | {accuracy} | {balanced_accuracy} | {macro_f1} | {weighted_f1} | {unknown_rate} | `{output_dir}` |\n".format(
                stage=record["stage"],
                mode=record["mode"],
                subset=record["subset"],
                n_samples=metrics.get("n_samples", 0),
                accuracy=format_metric(metrics.get("accuracy")),
                balanced_accuracy=format_metric(metrics.get("balanced_accuracy")),
                macro_f1=format_metric(metrics.get("macro_f1")),
                weighted_f1=format_metric(metrics.get("weighted_f1")),
                unknown_rate=format_metric(metrics.get("unknown_rate")),
                output_dir=record["output_dir"],
            )
        )

    lines.append("\n## Before vs After / 微调前后对比\n\n")
    for mode in MODES:
        for subset in SUBSET_TO_MANIFEST:
            before_record = index.get(("before", mode, subset))
            after_record = index.get(("after", mode, subset))
            if not before_record or not after_record:
                continue
            before_metrics = before_record["metrics"]
            after_metrics = after_record["metrics"]
            lines.append(f"### {mode} / {subset}\n\n")
            lines.append("| Metric | Before | After | Delta |\n")
            lines.append("|--------|--------|-------|-------|\n")
            for metric_name in REPORT_METRICS:
                before_value = before_metrics.get(metric_name)
                after_value = after_metrics.get(metric_name)
                delta = None
                if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                    delta = after_value - before_value
                lines.append(
                    f"| {metric_name} | {format_metric(before_value)} | {format_metric(after_value)} | {format_metric(delta)} |\n"
                )
            if mode == "3class":
                before_folded = before_metrics.get("folded_2class")
                after_folded = after_metrics.get("folded_2class")
                if before_folded and after_folded:
                    lines.append("\nFolded 2Class from 3Class / 三分类折叠成二分类\n\n")
                    lines.append("| Metric | Before | After | Delta |\n")
                    lines.append("|--------|--------|-------|-------|\n")
                    for metric_name in REPORT_METRICS:
                        before_value = before_folded.get(metric_name)
                        after_value = after_folded.get(metric_name)
                        delta = None
                        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                            delta = after_value - before_value
                        lines.append(
                            f"| folded::{metric_name} | {format_metric(before_value)} | {format_metric(after_value)} | {format_metric(delta)} |\n"
                        )
            lines.append("\n")

    lines.append("## 2Class vs 3Class / 二分类与三分类对比\n\n")
    for stage in STAGES:
        for subset in SUBSET_TO_MANIFEST:
            record_2c = index.get((stage, "2class", subset))
            record_3c = index.get((stage, "3class", subset))
            if not record_2c or not record_3c:
                continue
            metrics_2c = record_2c["metrics"]
            metrics_3c = record_3c["metrics"]
            folded_3c = metrics_3c.get("folded_2class", {})
            lines.append(f"### {stage} / {subset}\n\n")
            lines.append("| Metric | 2Class | 3Class Main | 3Class Folded 2Class | Delta(2Class-Folded) |\n")
            lines.append("|--------|--------|-------------|----------------------|----------------------|\n")
            for metric_name in REPORT_METRICS:
                value_2c = metrics_2c.get(metric_name)
                value_3c = metrics_3c.get(metric_name)
                value_folded = folded_3c.get(metric_name)
                delta = None
                if isinstance(value_2c, (int, float)) and isinstance(value_folded, (int, float)):
                    delta = value_2c - value_folded
                lines.append(
                    f"| {metric_name} | {format_metric(value_2c)} | {format_metric(value_3c)} | {format_metric(value_folded)} | {format_metric(delta)} |\n"
                )
            lines.append("\n")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[report] markdown: {report_md_path}")
    print(f"[report] json: {summary_json_path}")
    print(f"[report] csv: {summary_csv_path}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_stages = resolve_requested(args.stage, STAGES)
    requested_modes = resolve_requested(args.mode, MODES)
    requested_subsets = resolve_requested(args.subset, list(SUBSET_TO_MANIFEST.keys()))

    for stage in requested_stages:
        for mode in requested_modes:
            for subset in requested_subsets:
                run_single_eval(args, stage, mode, subset, repo_root)

    all_records = collect_available_results(run_dir)
    report_dir = run_dir / "evals" / "reports"
    write_report_files(all_records, report_dir, args)


if __name__ == "__main__":
    main()
