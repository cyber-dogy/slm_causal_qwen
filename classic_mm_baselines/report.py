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


def safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_std(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def load_run(run_root: Path) -> Dict[str, Any]:
    summary_path = run_root / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json: {summary_path}")

    payload = json.loads(summary_path.read_text())

    if isinstance(payload, list):
        config_candidates = sorted(run_root.glob("fold_*/config.json"))
        config = json.loads(config_candidates[0].read_text()) if config_candidates else {}
        return {
            "run_root": str(run_root),
            "experiment_name": config.get("experiment_name", config.get("fusion_type", run_root.name)),
            "task_mode": config.get("task_mode", "unknown"),
            "backbone_name": config.get("backbone_name", "resnet18"),
            "input_mode": config.get("input_mode", "triple_view"),
            "fusion_type": config.get("fusion_type", run_root.name),
            "rows": payload,
        }

    rows: List[Dict[str, Any]] = []
    for fold_result in payload["fold_results"]:
        for split_key, split_name in [("val_metrics", "val"), ("test_metrics", "test")]:
            metrics = fold_result[split_key]
            rows.append(
                {
                    "fold": fold_result["fold"],
                    "split": split_name,
                    "best_epoch": fold_result["best_epoch"],
                    "best_metric": fold_result["best_metric"],
                    "selection_metric_name": fold_result["selection_metric_name"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                }
            )

    return {
        "run_root": str(run_root),
        "experiment_name": payload.get("experiment_name", run_root.name),
        "task_mode": payload.get("task_mode", "unknown"),
        "backbone_name": payload.get("backbone_name", "resnet18"),
        "input_mode": payload.get("input_mode", "triple_view"),
        "fusion_type": payload.get("fusion_type", run_root.name),
        "rows": rows,
    }


def render_report(items: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    lines: List[str] = [
        "# Classic Multimodal Baseline Report / 经典多模态基线报告",
        "",
        "## Coverage / 覆盖实验",
        "",
        "| Experiment | Task | Backbone | Input | Fusion | Split | Folds | Accuracy | Balanced Acc | Macro F1 | Weighted F1 |",
        "|------------|------|----------|-------|--------|-------|-------|----------|--------------|----------|-------------|",
    ]

    aggregates: List[Dict[str, Any]] = []

    for item in items:
        for split in ["val", "test"]:
            rows = [row for row in item["rows"] if row["split"] == split]
            aggregate = {
                "experiment_name": item["experiment_name"],
                "task_mode": item["task_mode"],
                "backbone_name": item["backbone_name"],
                "input_mode": item["input_mode"],
                "fusion_type": item["fusion_type"],
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
                "run_root": item["run_root"],
            }
            aggregates.append(aggregate)
            lines.append(
                "| "
                f"{item['experiment_name']} | "
                f"{item['task_mode']} | "
                f"{item['backbone_name']} | "
                f"{item['input_mode']} | "
                f"{item['fusion_type']} | "
                f"{split} | "
                f"{aggregate['folds']} | "
                f"{aggregate['accuracy_mean']:.4f} ± {aggregate['accuracy_std']:.4f} | "
                f"{aggregate['balanced_accuracy_mean']:.4f} ± {aggregate['balanced_accuracy_std']:.4f} | "
                f"{aggregate['macro_f1_mean']:.4f} ± {aggregate['macro_f1_std']:.4f} | "
                f"{aggregate['weighted_f1_mean']:.4f} ± {aggregate['weighted_f1_std']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Notes / 说明",
            "",
            "- Input modes are read from the same data1 manifests and only differ in selected after-image views / 输入模式来自同一份 data1 manifest，只是选择的 after 视图不同。",
            "- `rgb_view1` means only the first RGB after-view / `rgb_view1` 表示仅使用第一路 RGB after 图像。",
            "- `single_view` always uses the first selected view, so `triple_view + single_view` is effectively `rgb_view1_after` / `single_view` 总是取当前输入模式的第一路视图，因此 `triple_view + single_view` 实际等价于只用 `rgb_view1_after`。",
            "- `rgb_dual` means `rgb_view1_after + rgb_view2_after` / `rgb_dual` 表示双 RGB after 输入。",
            "- `ir_only` means only `ir_after` / `ir_only` 表示仅使用 `ir_after`。",
            "- `late_fusion` performs per-view classification before weighted logit fusion / `late_fusion` 先做单视图分类，再做加权 logits 融合。",
            "",
        ]
    )
    return "\n".join(lines) + "\n", aggregates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate classic baseline CV runs into a report.")
    parser.add_argument("--run_roots", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = [load_run(Path(path).resolve()) for path in args.run_roots]
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
    print("Classic baseline report generated / 经典 baseline 报告已生成")
    print(f"Output dir / 输出目录: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
