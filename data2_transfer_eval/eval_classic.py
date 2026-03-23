#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classic_mm_baselines.dataset import (  # type: ignore
    CLASS_NAMES_2,
    CLASS_NAMES_3,
    ManifestMultimodalDataset,
    resolve_view_keys,
)
from classic_mm_baselines.metrics import compute_classification_metrics  # type: ignore
from classic_mm_baselines.models import MultiModalClassifier  # type: ignore


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ids": [item["id"] for item in batch],
        "images": torch.stack([item["images"] for item in batch], dim=0),
        "label_ids": torch.tensor([item["label_id"] for item in batch], dtype=torch.long),
        "label_names": [item["label_name"] for item in batch],
        "label_2c": [item["label_2c"] for item in batch],
        "label_3c": [item["label_3c"] for item in batch],
        "condition_uid": [item["condition_uid"] for item in batch],
        "role": [item["role"] for item in batch],
        "input_mode": batch[0]["input_mode"],
        "view_keys": batch[0]["view_keys"],
    }


def discover_run_roots(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.glob("official_cv_*")
        if path.is_dir() and not path.name.startswith("_smoke")
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fold_3class_to_2class(label: str) -> str:
    return "normal" if label == "normal" else "abnormal"


def infer_classic_defaults(run_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(config)
    name = run_root.name

    if "backbone_name" not in enriched:
        if "resnet50" in name:
            enriched["backbone_name"] = "resnet50"
        elif "resnet34" in name:
            enriched["backbone_name"] = "resnet34"
        elif "efficientnet_b0" in name:
            enriched["backbone_name"] = "efficientnet_b0"
        elif "vit_b_16" in name or "vit_b16" in name:
            enriched["backbone_name"] = "vit_b_16"
        else:
            enriched["backbone_name"] = "resnet18"

    if "input_mode" not in enriched:
        if "rgb_dual" in name:
            enriched["input_mode"] = "rgb_dual"
        elif "ir_only" in name:
            enriched["input_mode"] = "ir_only"
        elif "rgb_view1" in name:
            enriched["input_mode"] = "rgb_view1"
        elif "rgb_view2" in name:
            enriched["input_mode"] = "rgb_view2"
        else:
            enriched["input_mode"] = "triple_view"

    if "experiment_name" not in enriched:
        enriched["experiment_name"] = name

    return enriched


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_mode: str,
) -> Dict[str, Any]:
    class_names = CLASS_NAMES_2 if task_mode == "2class" else CLASS_NAMES_3
    criterion = nn.CrossEntropyLoss()
    model.eval()

    y_true: List[str] = []
    y_pred: List[str] = []
    y_prob: List[np.ndarray] = []
    predictions: List[Dict[str, Any]] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label_ids"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1

        for idx in range(images.shape[0]):
            true_label = class_names[int(labels[idx].item())]
            pred_label = class_names[int(preds[idx].item())]
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_prob.append(probs[idx].detach().cpu().numpy())
            predictions.append(
                {
                    "id": batch["ids"][idx],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "label_2c": batch["label_2c"][idx],
                    "label_3c": batch["label_3c"][idx],
                    "condition_uid": batch["condition_uid"][idx],
                    "role": batch["role"][idx],
                    "input_mode": batch["input_mode"],
                    "view_keys": list(batch["view_keys"]),
                }
            )

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        y_prob=np.stack(y_prob, axis=0) if y_prob else None,
        subset_name="data2_transfer",
    )
    metrics["loss"] = total_loss / max(1, n_batches)
    if task_mode == "3class":
        metrics["folded_2class"] = compute_classification_metrics(
            y_true=[fold_3class_to_2class(label) for label in y_true],
            y_pred=[fold_3class_to_2class(label) for label in y_pred],
            class_names=CLASS_NAMES_2,
            y_prob=None,
            subset_name="data2_transfer_folded_2class",
        )
    return {"metrics": metrics, "predictions": predictions}


def evaluate_run_root(
    run_root: Path,
    manifest_path: Path,
    output_root: Path,
    batch_size: int,
    num_workers: int,
    skip_existing: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for fold_dir in sorted(run_root.glob("fold_*")):
        config_path = fold_dir / "config.json"
        ckpt_path = fold_dir / "checkpoints" / "best.pt"
        if not config_path.exists() or not ckpt_path.exists():
            continue

        config = infer_classic_defaults(run_root, load_json(config_path))
        dataset = ManifestMultimodalDataset(
            manifest_path=manifest_path,
            target_mode=config["task_mode"],
            image_size=int(config.get("image_size", 224)),
            train=False,
            input_mode=config["input_mode"],
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalClassifier(
            backbone_name=config["backbone_name"],
            fusion_type=config["fusion_type"],
            num_classes=len(CLASS_NAMES_2 if config["task_mode"] == "2class" else CLASS_NAMES_3),
            num_views=len(resolve_view_keys(config["input_mode"])),
            pretrained=bool(config.get("pretrained", True)),
            dropout=float(config.get("dropout", 0.2)),
        ).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        output_dir = output_root / run_root.name / fold_dir.name / "after" / "transfer_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics_summary.json"

        if skip_existing and metrics_path.exists():
            metrics = load_json(metrics_path)
        else:
            result = evaluate_model(model=model, loader=loader, device=device, task_mode=config["task_mode"])
            write_json(metrics_path, result["metrics"])
            with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=result["predictions"][0].keys())
                writer.writeheader()
                writer.writerows(result["predictions"])
            metrics = result["metrics"]

        folded_macro_f1 = None
        if "folded_2class" in metrics:
            folded_macro_f1 = float(metrics["folded_2class"]["macro_f1"])
        records.append(
            {
                "family": "classic",
                "experiment_name": config.get("experiment_name", run_root.name),
                "task_mode": config["task_mode"],
                "method": "baseline",
                "backbone_name": config["backbone_name"],
                "input_mode": config["input_mode"],
                "fusion_type": config["fusion_type"],
                "stage": "after",
                "split": "transfer_test",
                "fold": fold_dir.name,
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "folded_2class_macro_f1": folded_macro_f1,
                "run_root": str(run_root),
                "output_dir": str(output_dir),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classic baselines on data2 transfer.")
    parser.add_argument("--manifest_path", type=str, default="/home/gjw/code/SLM_data/processed_qwen_data2_transfer/transfer_test_manifest.jsonl")
    parser.add_argument("--classic_runs_root", type=str, default="/home/gjw/code/Qwen-SLM/runs/classic_mm_baselines")
    parser.add_argument("--output_root", type=str, default="/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval/classic")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    runs_root = Path(args.classic_runs_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    for run_root in discover_run_roots(runs_root):
        print(f"[classic] evaluating {run_root.name}")
        all_records.extend(
            evaluate_run_root(
                run_root=run_root,
                manifest_path=manifest_path,
                output_root=output_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                skip_existing=args.skip_existing,
            )
        )

    write_json(output_root / "classic_summary.json", all_records)
    print("=" * 72)
    print("classic data2 transfer evaluation finished / classic data2 transfer 评估完成")
    print(f"Output / 输出: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
