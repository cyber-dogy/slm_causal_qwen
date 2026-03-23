#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Qwen2VLProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_vis_fusion.dataset import (  # type: ignore
    CLASS_NAMES_2,
    CLASS_NAMES_3,
    QwenVisFusionDataset,
    collate_fn,
    resolve_view_keys,
)
from qwen_vis_fusion.metrics import compute_classification_metrics  # type: ignore
from qwen_vis_fusion.model import QwenVisionFusionClassifier  # type: ignore


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_run_roots(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.glob("official_cv_*")
        if path.is_dir() and not path.name.startswith("_smoke")
    )


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {**batch, "label_ids": batch["label_ids"].to(device)}
    moved["pixel_values"] = batch["pixel_values"].to(device)
    moved["image_grid_thw"] = batch["image_grid_thw"].to(device)
    return moved


@torch.no_grad()
def evaluate_model(
    model: QwenVisionFusionClassifier,
    loader: DataLoader,
    device: torch.device,
    task_mode: str,
) -> Dict[str, Any]:
    class_names = CLASS_NAMES_2 if task_mode == "2class" else CLASS_NAMES_3
    model.eval()
    y_true: List[str] = []
    y_pred: List[str] = []
    y_prob: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits, _ = model(
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            sample_image_counts=batch["sample_image_counts"],
        )
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)

        for idx, pred_id in enumerate(pred_ids.tolist()):
            true_label = batch["label_names"][idx]
            pred_label = class_names[pred_id]
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_prob.append(probs[idx].detach().cpu().numpy())
            records.append(
                {
                    "sample_id": batch["sample_ids"][idx],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "label_2c": batch["labels_2c"][idx],
                    "label_3c": batch["labels_3c"][idx],
                    "condition_uid": batch["condition_uids"][idx],
                    "role": batch["roles"][idx],
                }
            )

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        y_prob=np.stack(y_prob, axis=0) if y_prob else None,
        subset_name="data2_transfer",
    )
    return {"metrics": metrics, "records": records}


def instantiate_model(config: Dict[str, Any]) -> QwenVisionFusionClassifier:
    class_names = CLASS_NAMES_2 if config["task_mode"] == "2class" else CLASS_NAMES_3
    return QwenVisionFusionClassifier(
        model_name_or_path=config["model_name_or_path"],
        num_views=len(resolve_view_keys(config["input_mode"])),
        num_classes=len(class_names),
        hidden_dim=int(config.get("hidden_dim", 512)),
        dropout=float(config.get("dropout", 0.1)),
        model_dtype=str(config.get("model_dtype", "float16")),
        unfreeze_last_n_blocks=int(config.get("unfreeze_last_n_blocks", 0)),
    )


def load_checkpoint_into_model(model: QwenVisionFusionClassifier, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)


def stage_specs_for_fold(run_root: Path, fold_dir: Path) -> List[Tuple[str, Dict[str, Any], Path]]:
    config = load_json(fold_dir / "config.json")
    specs: List[Tuple[str, Dict[str, Any], Path]] = []

    if int(config.get("unfreeze_last_n_blocks", 0)) > 0 and config.get("init_run_root"):
        init_run_root = Path(config["init_run_root"]).resolve()
        init_fold_dir = init_run_root / fold_dir.name
        init_config = load_json(init_fold_dir / "config.json")
        specs.append(("before", init_config, init_fold_dir / "best.pt"))

    specs.append(("after", config, fold_dir / "best.pt"))
    return specs


def evaluate_run_root(
    run_root: Path,
    manifest_path: Path,
    output_root: Path,
    batch_size: int,
    skip_existing: bool,
) -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []
    processor_cache: Dict[str, Qwen2VLProcessor] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_dir in sorted(run_root.glob("fold_*")):
        if not (fold_dir / "config.json").exists():
            continue

        for stage, config, checkpoint_path in stage_specs_for_fold(run_root, fold_dir):
            model_name = config["model_name_or_path"]
            if model_name not in processor_cache:
                processor_cache[model_name] = Qwen2VLProcessor.from_pretrained(model_name, trust_remote_code=True)
            processor = processor_cache[model_name]

            dataset = QwenVisFusionDataset(
                manifest_path=manifest_path,
                processor=processor,
                task_mode=config["task_mode"],
                input_mode=config["input_mode"],
                positive_seed=42,
                random_positive=False,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )

            out_dir = output_root / run_root.name / fold_dir.name / stage / "transfer_test"
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = out_dir / "metrics_summary.json"

            if skip_existing and metrics_path.exists():
                metrics = load_json(metrics_path)
            else:
                model = instantiate_model(config).to(device)
                load_checkpoint_into_model(model=model, checkpoint_path=checkpoint_path, device=device)
                result = evaluate_model(model=model, loader=loader, device=device, task_mode=config["task_mode"])
                metrics = result["metrics"]
                write_json(metrics_path, metrics)
                with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=result["records"][0].keys())
                    writer.writeheader()
                    writer.writerows(result["records"])
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            folded_macro_f1 = None
            if "folded_2class" in metrics:
                folded_macro_f1 = float(metrics["folded_2class"]["macro_f1"])

            all_records.append(
                {
                    "family": "qwen_vis",
                    "experiment_name": run_root.name,
                    "task_mode": config["task_mode"],
                    "method": config.get("method", "baseline"),
                    "backbone_name": "qwen2vl_vision",
                    "input_mode": config["input_mode"],
                    "fusion_type": "mlp_fusion",
                    "stage": stage,
                    "split": "transfer_test",
                    "fold": fold_dir.name,
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "weighted_f1": float(metrics["weighted_f1"]),
                    "folded_2class_macro_f1": folded_macro_f1,
                    "run_root": str(run_root),
                    "output_dir": str(out_dir),
                }
            )
    return all_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate qwen_vis_fusion runs on data2 transfer.")
    parser.add_argument("--manifest_path", type=str, default="/home/gjw/code/SLM_data/processed_qwen_data2_transfer/transfer_test_manifest.jsonl")
    parser.add_argument("--runs_root", type=str, default="/home/gjw/code/Qwen-SLM/runs/qwen_vis_fusion")
    parser.add_argument("--output_root", type=str, default="/home/gjw/code/Qwen-SLM/runs/data2_transfer_eval/qwen_vis")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    runs_root = Path(args.runs_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    for run_root in discover_run_roots(runs_root):
        print(f"[qwen_vis] evaluating {run_root.name}")
        all_records.extend(
            evaluate_run_root(
                run_root=run_root,
                manifest_path=manifest_path,
                output_root=output_root,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
            )
        )

    write_json(output_root / "qwen_vis_summary.json", all_records)
    print("=" * 72)
    print("qwen_vis data2 transfer evaluation finished / qwen_vis data2 transfer 评估完成")
    print(f"Output / 输出: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
