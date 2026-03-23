from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.path_utils import resolve_path


def safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_std(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def load_run(run_root: Path) -> Dict[str, Any]:
    payload = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for fold_result in payload["fold_results"]:
        metric_specs = [
            ("before_val_metrics", "before", "val"),
            ("before_test_metrics", "before", "test"),
            ("val_metrics", "after", "val"),
            ("test_metrics", "after", "test"),
        ]
        for split_key, stage_name, split_name in metric_specs:
            metrics = fold_result.get(split_key)
            if metrics is None:
                continue
            rows.append(
                {
                    "fold": fold_result["fold"],
                    "stage": stage_name,
                    "split": split_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "folded_2class_macro_f1": metrics.get("folded_2class", {}).get("macro_f1"),
                }
            )
    return {
        "experiment_name": payload["experiment_name"],
        "task_mode": payload["task_mode"],
        "method": payload["method"],
        "input_mode": payload["input_mode"],
        "unfreeze_last_n_blocks": payload.get("unfreeze_last_n_blocks", 0),
        "run_root": str(run_root),
        "rows": rows,
    }


def aggregate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregates: List[Dict[str, Any]] = []
    for item in items:
        for stage in sorted({row["stage"] for row in item["rows"]}):
            for split in ["val", "test"]:
                rows = [row for row in item["rows"] if row["split"] == split and row["stage"] == stage]
                if not rows:
                    continue
                aggregate = {
                    "experiment_name": item["experiment_name"],
                    "task_mode": item["task_mode"],
                    "method": item["method"],
                    "input_mode": item["input_mode"],
                    "unfreeze_last_n_blocks": item.get("unfreeze_last_n_blocks", 0),
                    "stage": stage,
                    "split": split,
                    "folds": len(rows),
                    "accuracy_mean": safe_mean([row["accuracy"] for row in rows]),
                    "accuracy_std": safe_std([row["accuracy"] for row in rows]),
                    "balanced_accuracy_mean": safe_mean([row["balanced_accuracy"] for row in rows]),
                    "balanced_accuracy_std": safe_std([row["balanced_accuracy"] for row in rows]),
                    "macro_f1_mean": safe_mean([row["macro_f1"] for row in rows]),
                    "macro_f1_std": safe_std([row["macro_f1"] for row in rows]),
                    "weighted_f1_mean": safe_mean([row["weighted_f1"] for row in rows]),
                    "weighted_f1_std": safe_std([row["weighted_f1"] for row in rows]),
                    "folded_2class_macro_f1_mean": safe_mean(
                        [row["folded_2class_macro_f1"] for row in rows if row["folded_2class_macro_f1"] is not None]
                    ),
                    "folded_2class_macro_f1_std": safe_std(
                        [row["folded_2class_macro_f1"] for row in rows if row["folded_2class_macro_f1"] is not None]
                    ),
                    "run_root": item["run_root"],
                }
                aggregates.append(aggregate)
    return aggregates


def load_best_classic_reference() -> Dict[Tuple[str, str], Dict[str, Any]]:
    path = REPO_ROOT / "outputs/classic_mm_baselines/reports/official_cv_extended/comparison_summary.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["task_mode"], row["split"])
        if key not in best or row["macro_f1_mean"] > best[key]["macro_f1_mean"]:
            best[key] = row
    return best


def load_prompt_qwen_reference() -> Dict[Tuple[str, str], Dict[str, Any]]:
    path = REPO_ROOT / "outputs/data1_protocol_reports/official_cv/comparison_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in payload.get("aggregates", []):
        if row.get("stage") != "after":
            continue
        key = (row["mode"], row["split"])
        current_macro_f1 = row.get("metrics", {}).get("macro_f1", {}).get("mean", 0.0)
        best_macro_f1 = best.get(key, {}).get("metrics", {}).get("macro_f1", {}).get("mean", -1.0)
        if current_macro_f1 > best_macro_f1:
            best[key] = row
    return best


def render_report(items: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    aggregates = aggregate_items(items)
    lines: List[str] = [
        "# Qwen Vision Fusion Report / Qwen 视觉融合报告",
        "",
        "## Coverage / 覆盖实验",
        "",
        "| Experiment | Task | Method | Input | Stage | Split | Folds | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Folded 2Class Macro F1 |",
        "|------------|------|--------|-------|-------|-------|-------|----------|--------------|----------|-------------|------------------------|",
    ]

    for row in aggregates:
        folded_text = "-"
        if row["folded_2class_macro_f1_mean"] > 0:
            folded_text = f"{row['folded_2class_macro_f1_mean']:.4f} ± {row['folded_2class_macro_f1_std']:.4f}"
        lines.append(
            "| "
            f"{row['experiment_name']} | {row['task_mode']} | {row['method']} | {row['input_mode']} | "
            f"{row['stage']} | "
            f"{row['split']} | {row['folds']} | "
            f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
            f"{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | "
            f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
            f"{row['weighted_f1_mean']:.4f} ± {row['weighted_f1_std']:.4f} | "
            f"{folded_text} |"
        )

    lines.extend(["", "## Before vs After / 微调前后对比", ""])
    for experiment_name in sorted({row["experiment_name"] for row in aggregates}):
        for split in ["val", "test"]:
            before = next((row for row in aggregates if row["experiment_name"] == experiment_name and row["split"] == split and row["stage"] == "before"), None)
            after = next((row for row in aggregates if row["experiment_name"] == experiment_name and row["split"] == split and row["stage"] == "after"), None)
            if before is None or after is None:
                continue
            lines.extend(
                [
                    f"### {experiment_name} / {split}",
                    "",
                    "| Metric | Before | After | Delta(After-Before) |",
                    "|--------|--------|-------|---------------------|",
                    f"| accuracy | {before['accuracy_mean']:.4f} | {after['accuracy_mean']:.4f} | {after['accuracy_mean'] - before['accuracy_mean']:.4f} |",
                    f"| balanced_accuracy | {before['balanced_accuracy_mean']:.4f} | {after['balanced_accuracy_mean']:.4f} | {after['balanced_accuracy_mean'] - before['balanced_accuracy_mean']:.4f} |",
                    f"| macro_f1 | {before['macro_f1_mean']:.4f} | {after['macro_f1_mean']:.4f} | {after['macro_f1_mean'] - before['macro_f1_mean']:.4f} |",
                    "",
                ]
            )

    lines.extend(["", "## Baseline vs Causal / Baseline 与 Causal 对比", ""])
    for task_mode in sorted({row["task_mode"] for row in aggregates}):
        for split in ["val", "test"]:
            baseline = next((row for row in aggregates if row["task_mode"] == task_mode and row["split"] == split and row["method"] == "baseline" and row["stage"] == "after" and row["unfreeze_last_n_blocks"] == 0), None)
            causal = next((row for row in aggregates if row["task_mode"] == task_mode and row["split"] == split and row["method"] == "causal" and row["stage"] == "after" and row["unfreeze_last_n_blocks"] == 0), None)
            if baseline is None or causal is None:
                continue
            lines.extend(
                [
                    f"### Frozen Head / {task_mode} / {split}",
                    "",
                    "| Metric | Baseline | Causal | Delta(Causal-Baseline) |",
                    "|--------|----------|--------|------------------------|",
                    f"| accuracy | {baseline['accuracy_mean']:.4f} | {causal['accuracy_mean']:.4f} | {causal['accuracy_mean'] - baseline['accuracy_mean']:.4f} |",
                    f"| balanced_accuracy | {baseline['balanced_accuracy_mean']:.4f} | {causal['balanced_accuracy_mean']:.4f} | {causal['balanced_accuracy_mean'] - baseline['balanced_accuracy_mean']:.4f} |",
                    f"| macro_f1 | {baseline['macro_f1_mean']:.4f} | {causal['macro_f1_mean']:.4f} | {causal['macro_f1_mean'] - baseline['macro_f1_mean']:.4f} |",
                    "",
                ]
            )
            unfrozen_baseline = next((row for row in aggregates if row["task_mode"] == task_mode and row["split"] == split and row["method"] == "baseline" and row["stage"] == "after" and row["unfreeze_last_n_blocks"] > 0), None)
            unfrozen_causal = next((row for row in aggregates if row["task_mode"] == task_mode and row["split"] == split and row["method"] == "causal" and row["stage"] == "after" and row["unfreeze_last_n_blocks"] > 0), None)
            if unfrozen_baseline is not None or unfrozen_causal is not None:
                lines.extend(
                    [
                        f"### Unfreeze Last Blocks / {task_mode} / {split}",
                        "",
                        "| Variant | Accuracy | Balanced Acc | Macro F1 |",
                        "|---------|----------|--------------|----------|",
                    ]
                )
                if baseline is not None:
                    lines.append(f"| frozen_baseline | {baseline['accuracy_mean']:.4f} | {baseline['balanced_accuracy_mean']:.4f} | {baseline['macro_f1_mean']:.4f} |")
                if causal is not None:
                    lines.append(f"| frozen_causal | {causal['accuracy_mean']:.4f} | {causal['balanced_accuracy_mean']:.4f} | {causal['macro_f1_mean']:.4f} |")
                if unfrozen_baseline is not None:
                    lines.append(f"| unfrozen_baseline | {unfrozen_baseline['accuracy_mean']:.4f} | {unfrozen_baseline['balanced_accuracy_mean']:.4f} | {unfrozen_baseline['macro_f1_mean']:.4f} |")
                if unfrozen_causal is not None:
                    lines.append(f"| unfrozen_causal | {unfrozen_causal['accuracy_mean']:.4f} | {unfrozen_causal['balanced_accuracy_mean']:.4f} | {unfrozen_causal['macro_f1_mean']:.4f} |")
                lines.append("")

    classic_best = load_best_classic_reference()
    prompt_qwen = load_prompt_qwen_reference()
    if classic_best or prompt_qwen:
        lines.extend(["## External Comparison / 外部对照", ""])
        lines.extend(
            [
                "| Task | Split | Current Best QwenVisFusion | Macro F1 | Prompt-Qwen After | Macro F1 | Best Classic | Macro F1 |",
                "|------|-------|---------------------------|----------|-------------------|----------|--------------|----------|",
            ]
        )
        for task_mode in ["2class", "3class"]:
            for split in ["val", "test"]:
                current_rows = [row for row in aggregates if row["task_mode"] == task_mode and row["split"] == split]
                if not current_rows:
                    continue
                current_best = max(
                    [row for row in current_rows if row["stage"] == "after"],
                    key=lambda item: item["macro_f1_mean"],
                )
                prompt_row = prompt_qwen.get((task_mode, split), {})
                classic_row = classic_best.get((task_mode, split), {})
                lines.append(
                    "| "
                    f"{task_mode} | {split} | {current_best['experiment_name']} ({current_best['method']}) | {current_best['macro_f1_mean']:.4f} | "
                    f"{prompt_row.get('experiment_label', 'after')} | {prompt_row.get('metrics', {}).get('macro_f1', {}).get('mean', prompt_row.get('metrics', {}).get('macro_f1', 0.0)):.4f} | "
                    f"{classic_row.get('experiment_name', '-')} | {classic_row.get('macro_f1_mean', 0.0):.4f} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Notes / 说明",
            "",
            "- This branch uses Qwen2-VL as a vision encoder and trains a discriminative MLP fusion head / 该分支将 Qwen2-VL 作为视觉编码器，并训练判别式 MLP 融合头。",
            "- The current official configs freeze the Qwen vision tower and train the fusion head only / 当前 official 配置默认冻结 Qwen 视觉塔，仅训练融合头。",
            "- Some fine-tune runs initialize from the frozen-head checkpoint, so `before` directly means the checkpoint before true visual fine-tuning / 部分真微调实验会从冻结版 checkpoint 初始化，因此报告中的 `before` 直接对应视觉真微调前的基线 checkpoint。",
            "- `baseline` uses classification loss only / `baseline` 仅使用分类损失。",
            "- `causal` adds consistency over fused visual representations for same-label different-condition pairs / `causal` 在融合视觉表征上对同标签不同 condition 正样本加入一致性约束。",
            "",
        ]
    )
    return "\n".join(lines) + "\n", aggregates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Qwen vision fusion official_cv runs.")
    parser.add_argument("--run_roots", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = [load_run(resolve_path(path)) for path in args.run_roots]
    report_text, aggregates = render_report(items)

    (output_dir / "comparison_report.md").write_text(report_text, encoding="utf-8")
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(aggregates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregates[0].keys())
        writer.writeheader()
        writer.writerows(aggregates)

    print("=" * 72)
    print("Qwen vision fusion report generated / Qwen 视觉融合报告已生成")
    print(f"Output dir / 输出目录: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
