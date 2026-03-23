#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPORT_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "folded_2class_macro_f1"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_std(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def format_mean_std(values: List[float]) -> str:
    if not values:
        return "-"
    return f"{safe_mean(values):.4f} ± {safe_std(values):.4f}"


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["family"],
            row["experiment_name"],
            row["task_mode"],
            row["method"],
            row["input_mode"],
            row["stage"],
            row["split"],
        )
        grouped[key].append(row)

    aggregates: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        family, experiment_name, task_mode, method, input_mode, stage, split = key
        aggregate = {
            "family": family,
            "experiment_name": experiment_name,
            "task_mode": task_mode,
            "method": method,
            "input_mode": input_mode,
            "stage": stage,
            "split": split,
            "folds": len(items),
            "backbone_name": items[0].get("backbone_name", ""),
            "fusion_type": items[0].get("fusion_type", ""),
            "run_root": items[0].get("run_root", ""),
        }
        for metric in REPORT_METRICS:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            aggregate[f"{metric}_mean"] = safe_mean(values)
            aggregate[f"{metric}_std"] = safe_std(values)
        aggregates.append(aggregate)
    return aggregates


def best_row(aggregates: List[Dict[str, Any]], *, family: str | None = None, task_mode: str, stage: str = "after") -> Dict[str, Any] | None:
    candidates = [
        row for row in aggregates
        if row["task_mode"] == task_mode and row["split"] == "transfer_test" and row["stage"] == stage
        and (family is None or row["family"] == family)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["macro_f1_mean"])


def family_label(name: str) -> str:
    mapping = {
        "prompt_qwen": "Prompt-Qwen",
        "classic": "Classic",
        "qwen_vis": "QwenVisFusion",
    }
    return mapping.get(name, name)


def render_report(aggregates: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Data2 Transfer Comparison Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Data2 Overview",
        "",
        "- `data2` is treated as a harder external transfer setting with unseen model/process shifts.",
        "- `data1` remains the condition-heldout closed-set source protocol.",
        "",
        "| Domain | Samples | Conditions | normal | HEW | LEL |",
        "|--------|---------|------------|--------|-----|-----|",
        f"| data1 | {analysis['source']['n_samples']} | {analysis['source']['n_conditions']} | {analysis['source']['label_counts'].get('normal', 0)} | {analysis['source']['label_counts'].get('HEW', 0)} | {analysis['source']['label_counts'].get('LEL', 0)} |",
        f"| data2 | {analysis['target']['n_samples']} | {analysis['target']['n_conditions']} | {analysis['target']['label_counts'].get('normal', 0)} | {analysis['target']['label_counts'].get('HEW', 0)} | {analysis['target']['label_counts'].get('LEL', 0)} |",
        "",
        "## Coverage",
        "",
        "| Family | Experiment | Task | Method | Input | Stage | Folds | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Folded 2Class Macro F1 |",
        "|--------|------------|------|--------|-------|-------|-------|----------|--------------|----------|-------------|------------------------|",
    ]

    for row in aggregates:
        folded_cell = "-"
        if row["folded_2class_macro_f1_mean"] > 0:
            folded_cell = (
                f"{row['folded_2class_macro_f1_mean']:.4f} ± "
                f"{row['folded_2class_macro_f1_std']:.4f}"
            )
        lines.append(
            "| "
            f"{family_label(row['family'])} | "
            f"{row['experiment_name']} | "
            f"{row['task_mode']} | "
            f"{row['method']} | "
            f"{row['input_mode']} | "
            f"{row['stage']} | "
            f"{row['folds']} | "
            f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
            f"{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | "
            f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
            f"{row['weighted_f1_mean']:.4f} ± {row['weighted_f1_std']:.4f} | "
            f"{folded_cell} |"
        )

    lines.extend(["", "## Best By Family", "", "| Task | Prompt-Qwen | Classic | QwenVisFusion |", "|------|-------------|---------|---------------|"])
    for task_mode in ["2class", "3class"]:
        prompt_best = best_row(aggregates, family="prompt_qwen", task_mode=task_mode)
        classic_best = best_row(aggregates, family="classic", task_mode=task_mode)
        qwen_best = best_row(aggregates, family="qwen_vis", task_mode=task_mode)

        def cell(row: Dict[str, Any] | None) -> str:
            if not row:
                return "-"
            return f"`{row['experiment_name']}` / {row['macro_f1_mean']:.4f}"

        lines.append(f"| {task_mode} | {cell(prompt_best)} | {cell(classic_best)} | {cell(qwen_best)} |")

    lines.extend(["", "## Before vs After", ""])
    for experiment_name in [
        "data1_official_cv_2class_baseline",
        "data1_official_cv_3class_baseline",
        "data1_official_cv_3class_causal",
        "official_cv_3class_rgb_dual_last2blocks_baseline",
        "official_cv_3class_rgb_dual_last2blocks_causal",
    ]:
        current = [row for row in aggregates if row["experiment_name"] == experiment_name and row["split"] == "transfer_test"]
        before = next((row for row in current if row["stage"] == "before"), None)
        after = next((row for row in current if row["stage"] == "after"), None)
        if not before or not after:
            continue
        lines.extend(
            [
                f"### {experiment_name}",
                "",
                "| Metric | Before | After | Delta(After-Before) |",
                "|--------|--------|-------|---------------------|",
                f"| accuracy | {before['accuracy_mean']:.4f} | {after['accuracy_mean']:.4f} | {after['accuracy_mean'] - before['accuracy_mean']:.4f} |",
                f"| balanced_accuracy | {before['balanced_accuracy_mean']:.4f} | {after['balanced_accuracy_mean']:.4f} | {after['balanced_accuracy_mean'] - before['balanced_accuracy_mean']:.4f} |",
                f"| macro_f1 | {before['macro_f1_mean']:.4f} | {after['macro_f1_mean']:.4f} | {after['macro_f1_mean'] - before['macro_f1_mean']:.4f} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- `data2` is a harder transfer setting than the source-side `official_cv`; its scores should be interpreted as external stress-test results.",
            "- Prompt-Qwen rows use the original prompt-based adapter evaluation pipeline on the new `data2` transfer manifest.",
            "- Classic rows reuse trained `data1 official_cv` checkpoints and only change the evaluation manifest.",
            "- QwenVisFusion rows reuse trained discriminative checkpoints; `last2blocks` runs include both `before` and `after` stages when available.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_bilingual_note() -> str:
    return "\n".join(
        [
            "# Data2 Transfer Report Note / data2 迁移报告说明",
            "",
            "- This report evaluates all previously trained families on the same `data2 transfer_test` manifest. / 本报告将此前训练好的各模型家族统一迁移到同一份 `data2 transfer_test` 清单上评估。",
            "- `before` vs `after` is only available for model families where a pre-adaptation or pre-finetuning checkpoint is meaningful. / `before` 与 `after` 仅对存在可比较预适配或预微调 checkpoint 的模型家族成立。",
            "- `Macro F1` is the primary metric for 3-class transfer. / `Macro F1` 是三分类迁移结果的主指标。",
            "- `folded_2class_macro_f1` folds `HEW/LEL` into `abnormal`. / `folded_2class_macro_f1` 将 `HEW/LEL` 折叠为 `abnormal`。",
            "",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate data2 transfer evaluations into one report.")
    parser.add_argument("--analysis_json", type=str, default="/home/gjw/code/SLM_data/processed_qwen_data2_transfer/data2_analysis.json")
    parser.add_argument("--prompt_summary", type=str, default="/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval/prompt_qwen/prompt_qwen_summary.json")
    parser.add_argument("--classic_summary", type=str, default="/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval/classic/classic_summary.json")
    parser.add_argument("--qwen_vis_summary", type=str, default="/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval/qwen_vis/qwen_vis_summary.json")
    parser.add_argument("--output_dir", type=str, default="/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval/reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = load_json(Path(args.analysis_json))
    all_rows: List[Dict[str, Any]] = []
    for summary_path in [args.prompt_summary, args.classic_summary, args.qwen_vis_summary]:
        path = Path(summary_path)
        if path.exists():
            all_rows.extend(load_json(path))

    aggregates = aggregate_rows(all_rows)
    report_text = render_report(aggregates=aggregates, analysis=analysis)
    bilingual_text = report_text + "\n" + render_bilingual_note()

    (output_dir / "comparison_report.md").write_text(report_text, encoding="utf-8")
    (output_dir / "comparison_report.zh_en.md").write_text(bilingual_text, encoding="utf-8")
    (output_dir / "comparison_summary.json").write_text(json.dumps(aggregates, indent=2, ensure_ascii=False), encoding="utf-8")

    if aggregates:
        with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregates[0].keys()))
            writer.writeheader()
            writer.writerows(aggregates)

    print("=" * 72)
    print("data2 transfer report generated / data2 transfer 报告已生成")
    print(f"Output / 输出: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
