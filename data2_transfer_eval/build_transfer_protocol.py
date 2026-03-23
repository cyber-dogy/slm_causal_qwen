#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_qwen2vl_slm import (  # type: ignore
    _clone_sample_with_split,
    _load_samples_from_dataset_dir,
    _save_manifest_bundle,
    _validate_split_integrity,
)


def load_metadata_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def counter_to_rows(counter: Counter) -> List[Dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common()]


def dataset_overview(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    condition_ids = [row["condition_id"] for row in rows]
    label_counts = Counter(row["defect_label_type"] for row in rows)
    role_counts = Counter(row["role"] for row in rows)
    family_counts = Counter(row["condition_id"].split("-")[0] for row in rows)
    strategy_counts = Counter(row.get("scan_strategy", "") for row in rows)
    power_values = [float(row["power_w"]) for row in rows if row.get("power_w")]
    speed_values = [float(row["speed_mms"]) for row in rows if row.get("speed_mms")]
    spacing_values = [float(row["spacing_mm"]) for row in rows if row.get("spacing_mm")]

    return {
        "n_samples": len(rows),
        "n_conditions": len(set(condition_ids)),
        "label_counts": dict(label_counts),
        "role_counts": dict(role_counts),
        "condition_family_counts": dict(family_counts),
        "scan_strategy_counts": dict(strategy_counts),
        "power_range_w": [min(power_values), max(power_values)] if power_values else [],
        "speed_range_mms": [min(speed_values), max(speed_values)] if speed_values else [],
        "spacing_range_mm": [min(spacing_values), max(spacing_values)] if spacing_values else [],
    }


def compare_datasets(source_rows: List[Dict[str, Any]], target_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def families(rows: List[Dict[str, Any]]) -> Counter:
        return Counter(row["condition_id"].split("-")[0] for row in rows)

    return {
        "source": dataset_overview(source_rows),
        "target": dataset_overview(target_rows),
        "notes": [
            "data2 被视为未见模型/未见工艺的 harder transfer 场景，而不是此前单纯的跨工况测试。",
            "data2 与 data1 在角色分布、条件家族分布和样本规模上均存在明显差异。",
            "因此 data2 结果应解释为更强外部迁移压力测试，而不是与 data1 official_cv 同难度的闭集验证。",
        ],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_analysis_md(comparison: Dict[str, Any], manifest_bundle: Dict[str, str]) -> str:
    source = comparison["source"]
    target = comparison["target"]

    lines = [
        "# Data2 Transfer Analysis",
        "",
        "## Positioning / 定位",
        "",
        "- `data1` 继续作为 source-domain 条件级闭集评估协议。",
        "- `data2` 现在被定义为更困难的外部 transfer 场景：包含未见模型与未见工艺的复合迁移压力，而不只是此前的跨工况。",
        "- 本目录只构建 `transfer_test`，不参与 data2 内部训练/验证切分。",
        "",
        "## Protocol Files / 协议文件",
        "",
        f"- Transfer manifest: `{manifest_bundle['transfer_test_manifest']}`",
        f"- Split summary: `{manifest_bundle['split_summary']}`",
        f"- Manifest all: `{manifest_bundle['manifest_all']}`",
        "",
        "## Source vs Target Overview / 源域与目标域概览",
        "",
        "| Domain | Samples | Conditions | normal | HEW | LEL |",
        "|--------|---------|------------|--------|-----|-----|",
        f"| data1 | {source['n_samples']} | {source['n_conditions']} | {source['label_counts'].get('normal', 0)} | {source['label_counts'].get('HEW', 0)} | {source['label_counts'].get('LEL', 0)} |",
        f"| data2 | {target['n_samples']} | {target['n_conditions']} | {target['label_counts'].get('normal', 0)} | {target['label_counts'].get('HEW', 0)} | {target['label_counts'].get('LEL', 0)} |",
        "",
        "## Data2 Roles / data2 角色分布",
        "",
        "| Role | Count |",
        "|------|-------|",
    ]

    for role, count in Counter(target["role_counts"]).items():
        lines.append(f"| `{role}` | {count} |")

    lines.extend(
        [
            "",
            "## Data2 Condition Families / data2 条件家族分布",
            "",
            "| Family | Count |",
            "|--------|-------|",
        ]
    )
    for family, count in Counter(target["condition_family_counts"]).items():
        lines.append(f"| `{family}` | {count} |")

    lines.extend(
        [
            "",
            "## Data2 Parameter Ranges / data2 参数范围",
            "",
            f"- Power / 功率: `{target['power_range_w']}` W",
            f"- Speed / 速度: `{target['speed_range_mms']}` mm/s",
            f"- Spacing / 间距: `{target['spacing_range_mm']}` mm",
            "",
            "## Notes / 说明",
            "",
        ]
    )
    for note in comparison["notes"]:
        lines.append(f"- {note}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an isolated data2 transfer protocol manifest.")
    parser.add_argument("--data1_dir", type=str, default="/home/gjw/code/SLM_data/Causal_Image_Data")
    parser.add_argument("--data2_dir", type=str, default="/home/gjw/code/SLM_data/Causal_Image_Data2")
    parser.add_argument("--output_root", type=str, default="/home/gjw/code/SLM_data/processed_qwen_data2_transfer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data1_dir = Path(args.data1_dir).resolve()
    data2_dir = Path(args.data2_dir).resolve()
    output_root = Path(args.output_root).resolve()

    source_rows = load_metadata_rows(data1_dir / "metadata.csv")
    target_rows = load_metadata_rows(data2_dir / "metadata.csv")

    target_samples = _load_samples_from_dataset_dir(data2_dir, dataset_domain="data2")
    transfer_test = [_clone_sample_with_split(sample, "transfer_test") for sample in target_samples]
    _validate_split_integrity(
        split_name="transfer_test",
        samples=transfer_test,
        expected_domain="data2",
        require_full_3class=True,
    )

    bundle = _save_manifest_bundle(
        split_map={"transfer_test": transfer_test},
        output_root=output_root,
        config={
            "builder": "data2_transfer_eval/build_transfer_protocol.py",
            "data1_dir": str(data1_dir),
            "data2_dir": str(data2_dir),
            "task": "data2_transfer_only",
        },
    )

    comparison = compare_datasets(source_rows=source_rows, target_rows=target_rows)
    write_json(output_root / "data2_analysis.json", comparison)
    write_csv(
        output_root / "data2_role_summary.csv",
        [{"role": name, "count": count} for name, count in Counter(target_rows[i]["role"] for i in range(len(target_rows))).most_common()],
    )
    write_csv(
        output_root / "data2_condition_summary.csv",
        [
            {
                "condition_id": condition_id,
                "family": condition_id.split("-")[0],
                "count": count,
            }
            for condition_id, count in Counter(row["condition_id"] for row in target_rows).most_common()
        ],
    )
    (output_root / "data2_analysis.md").write_text(
        render_analysis_md(comparison=comparison, manifest_bundle=bundle),
        encoding="utf-8",
    )

    print("=" * 72)
    print("data2 transfer protocol built / data2 transfer 协议已生成")
    print(f"Output root / 输出目录: {output_root}")
    print(f"Manifest / 清单: {bundle['transfer_test_manifest']}")
    print("=" * 72)


if __name__ == "__main__":
    main()

