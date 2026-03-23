from __future__ import annotations

import argparse
import csv
import json
import random
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

from common.path_utils import expand_env_placeholders, load_json_config, resolve_path
from classic_mm_baselines.dataset import (
    CLASS_NAMES_2,
    CLASS_NAMES_3,
    ManifestMultimodalDataset,
    available_input_modes,
    resolve_view_keys,
)
from classic_mm_baselines.metrics import compute_classification_metrics
from classic_mm_baselines.models import MultiModalClassifier


BACKBONE_CHOICES = ["resnet18", "resnet34", "resnet50", "efficientnet_b0", "vit_b_16"]
FUSION_CHOICES = ["single_view", "mean_pool", "mlp_concat", "gated_mlp", "cross_attention", "late_fusion"]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool value: {value}")


def load_json(path: str | Path) -> Dict[str, Any]:
    return load_json_config(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def available_folds(protocol_root: Path, scope: str) -> List[str]:
    if scope == "quick_dev":
        return ["quick_dev"]
    scope_root = protocol_root / "official_cv"
    return sorted(path.name for path in scope_root.iterdir() if path.is_dir())


def manifest_path(protocol_root: Path, scope: str, fold: str, split: str) -> Path:
    if scope == "quick_dev":
        return protocol_root / "quick_dev" / f"{split}_manifest.jsonl"
    return protocol_root / "official_cv" / fold / f"{split}_manifest.jsonl"


def fold_run_dir(run_root: Path, scope: str, fold: str) -> Path:
    return run_root if scope == "quick_dev" else run_root / fold


def class_names_for_mode(task_mode: str) -> List[str]:
    return CLASS_NAMES_2 if task_mode == "2class" else CLASS_NAMES_3


def selection_metric_name(task_mode: str) -> str:
    return "balanced_accuracy" if task_mode == "2class" else "macro_f1"


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.num_epochs))
    return None


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ids": [item["id"] for item in batch],
        "images": torch.stack([item["images"] for item in batch], dim=0),
        "view_keys": batch[0]["view_keys"],
        "input_mode": batch[0]["input_mode"],
        "label_ids": torch.tensor([item["label_id"] for item in batch], dtype=torch.long),
        "label_names": [item["label_name"] for item in batch],
        "label_2c": [item["label_2c"] for item in batch],
        "label_3c": [item["label_3c"] for item in batch],
        "condition_uid": [item["condition_uid"] for item in batch],
        "role": [item["role"] for item in batch],
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_mode: str,
    criterion: nn.Module,
    output_dir: Path,
    subset_name: str,
) -> Dict[str, Any]:
    model.eval()
    class_names = class_names_for_mode(task_mode)

    y_true_names: List[str] = []
    y_pred_names: List[str] = []
    y_prob: List[np.ndarray] = []
    predictions: List[Dict[str, Any]] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label_ids"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        total_loss += float(loss.item())
        n_batches += 1

        for idx in range(images.shape[0]):
            pred_id = int(preds[idx].item())
            true_id = int(labels[idx].item())
            y_true_names.append(class_names[true_id])
            y_pred_names.append(class_names[pred_id])
            y_prob.append(probs[idx].detach().cpu().numpy())
            predictions.append(
                {
                    "id": batch["ids"][idx],
                    "true_label": class_names[true_id],
                    "pred_label": class_names[pred_id],
                    "condition_uid": batch["condition_uid"][idx],
                    "role": batch["role"][idx],
                    "input_mode": batch["input_mode"],
                    "view_keys": list(batch["view_keys"]),
                    "probabilities": {
                        class_names[j]: float(probs[idx, j].item()) for j in range(len(class_names))
                    },
                }
            )

    metrics = compute_classification_metrics(
        y_true=y_true_names,
        y_pred=y_pred_names,
        class_names=class_names,
        y_prob=np.stack(y_prob, axis=0) if y_prob else None,
        subset_name=subset_name,
    )
    metrics["loss"] = total_loss / max(1, n_batches)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


def train_one_fold(
    args: argparse.Namespace,
    protocol_root: Path,
    fold_name: str,
) -> Dict[str, Any]:
    run_dir = fold_run_dir(Path(args.run_root), args.scope, fold_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "evals").mkdir(exist_ok=True)

    train_ds = ManifestMultimodalDataset(
        manifest_path=manifest_path(protocol_root, args.scope, fold_name, "train"),
        target_mode=args.task_mode,
        image_size=args.image_size,
        train=True,
        input_mode=args.input_mode,
    )
    val_ds = ManifestMultimodalDataset(
        manifest_path=manifest_path(protocol_root, args.scope, fold_name, "val"),
        target_mode=args.task_mode,
        image_size=args.image_size,
        train=False,
        input_mode=args.input_mode,
    )
    test_ds = ManifestMultimodalDataset(
        manifest_path=manifest_path(protocol_root, args.scope, fold_name, "test"),
        target_mode=args.task_mode,
        image_size=args.image_size,
        train=False,
        input_mode=args.input_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    num_classes = len(class_names_for_mode(args.task_mode))
    num_views = len(resolve_view_keys(args.input_mode))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalClassifier(
        backbone_name=args.backbone_name,
        fusion_type=args.fusion_type,
        num_classes=num_classes,
        num_views=num_views,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    optimizer = build_optimizer(args=args, model=model)
    scheduler = build_scheduler(args=args, optimizer=optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    selection_metric = selection_metric_name(args.task_mode)
    best_metric = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_log: List[Dict[str, Any]] = []
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["label_ids"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * images.shape[0]
            total_items += images.shape[0]

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / max(1, total_items)
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            task_mode=args.task_mode,
            criterion=criterion,
            output_dir=run_dir / "evals" / "val_latest",
            subset_name=f"{fold_name}_val",
        )
        val_metric = float(val_metrics[selection_metric])
        train_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "selection_metric_name": selection_metric,
                "val_selection_metric": val_metric,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "selection_metric_name": selection_metric,
                },
                run_dir / "checkpoints" / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            break

    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_val_metrics = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        task_mode=args.task_mode,
        criterion=criterion,
        output_dir=run_dir / "evals" / "val",
        subset_name=f"{fold_name}_val",
    )
    best_test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        task_mode=args.task_mode,
        criterion=criterion,
        output_dir=run_dir / "evals" / "test",
        subset_name=f"{fold_name}_test",
    )

    config = {
        **vars(args),
        "protocol_root": str(protocol_root),
        "run_dir": str(run_dir),
        "train_manifest": str(manifest_path(protocol_root, args.scope, fold_name, "train")),
        "val_manifest": str(manifest_path(protocol_root, args.scope, fold_name, "val")),
        "test_manifest": str(manifest_path(protocol_root, args.scope, fold_name, "test")),
        "view_keys": resolve_view_keys(args.input_mode),
        "selection_metric_name": selection_metric,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with (run_dir / "logs" / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "selection_metric_name": selection_metric,
                "logs": train_log,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    if train_log:
        with (run_dir / "logs" / "train_log.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=train_log[0].keys())
            writer.writeheader()
            writer.writerows(train_log)

    return {
        "fold": fold_name,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "selection_metric_name": selection_metric,
        "val_metrics": best_val_metrics,
        "test_metrics": best_test_metrics,
    }


def write_run_summary(args: argparse.Namespace, results: List[Dict[str, Any]], output_path: Path) -> None:
    payload = {
        "experiment_name": args.experiment_name,
        "task_mode": args.task_mode,
        "backbone_name": args.backbone_name,
        "input_mode": args.input_mode,
        "fusion_type": args.fusion_type,
        "scope": args.scope,
        "run_root": args.run_root,
        "fold_results": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    known, _ = config_parser.parse_known_args()
    config_defaults = load_json(known.config) if known.config else {}

    def cfg(name: str, fallback):
        return config_defaults.get(name, fallback)

    parser = argparse.ArgumentParser(description="Train classic multimodal baselines on the current data1 protocol.")
    parser.set_defaults(**config_defaults)
    parser.add_argument("--config", type=str, default=known.config)
    parser.add_argument("--protocol_root", type=str, default=cfg("protocol_root", "${SLM_DATA1_PROTOCOL_ROOT}"))
    parser.add_argument("--scope", type=str, default=cfg("scope", "official_cv"), choices=["official_cv", "quick_dev"])
    parser.add_argument("--task_mode", type=str, default=cfg("task_mode", None), choices=["2class", "3class"])
    parser.add_argument("--backbone_name", type=str, default=cfg("backbone_name", "resnet18"), choices=BACKBONE_CHOICES)
    parser.add_argument("--input_mode", type=str, default=cfg("input_mode", "triple_view"), choices=available_input_modes())
    parser.add_argument("--fusion_type", type=str, default=cfg("fusion_type", None), choices=FUSION_CHOICES)
    parser.add_argument("--experiment_name", type=str, default=cfg("experiment_name", None))
    parser.add_argument("--run_root", type=str, default=cfg("run_root", None))
    parser.add_argument("--fold", type=str, default=cfg("fold", "all"))
    parser.add_argument("--batch_size", type=int, default=cfg("batch_size", 16))
    parser.add_argument("--num_epochs", type=int, default=cfg("num_epochs", 20))
    parser.add_argument("--early_stopping_patience", type=int, default=cfg("early_stopping_patience", 5))
    parser.add_argument("--learning_rate", type=float, default=cfg("learning_rate", 1e-4))
    parser.add_argument("--weight_decay", type=float, default=cfg("weight_decay", 1e-4))
    parser.add_argument("--dropout", type=float, default=cfg("dropout", 0.2))
    parser.add_argument("--image_size", type=int, default=cfg("image_size", 224))
    parser.add_argument("--num_workers", type=int, default=cfg("num_workers", 4))
    parser.add_argument("--seed", type=int, default=cfg("seed", 42))
    parser.add_argument("--pretrained", type=str2bool, default=cfg("pretrained", True))
    parser.add_argument("--optimizer", type=str, default=cfg("optimizer", "adamw"), choices=["adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default=cfg("scheduler", "cosine"), choices=["none", "cosine"])
    parser.add_argument("--momentum", type=float, default=cfg("momentum", 0.9))
    parser.add_argument("--label_smoothing", type=float, default=cfg("label_smoothing", 0.0))
    parser.add_argument("--amp", type=str2bool, default=cfg("amp", True))
    args = parser.parse_args()

    if not args.task_mode:
        raise ValueError("--task_mode is required")
    if not args.fusion_type:
        raise ValueError("--fusion_type is required")
    if not args.run_root:
        raise ValueError("--run_root is required")
    if args.experiment_name is None:
        args.experiment_name = f"{args.backbone_name}_{args.input_mode}_{args.fusion_type}_{args.task_mode}"
    args.protocol_root = expand_env_placeholders(args.protocol_root)
    args.run_root = expand_env_placeholders(args.run_root)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    protocol_root = resolve_path(args.protocol_root)
    run_root = resolve_path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    folds = available_folds(protocol_root, args.scope)
    if args.fold != "all":
        folds = [args.fold]

    all_results = []
    for fold in folds:
        print("=" * 72)
        print(
            "Training classic baseline: "
            f"exp={args.experiment_name}, task={args.task_mode}, "
            f"backbone={args.backbone_name}, input={args.input_mode}, "
            f"fusion={args.fusion_type}, fold={fold}"
        )
        print("=" * 72)
        result = train_one_fold(args=args, protocol_root=protocol_root, fold_name=fold)
        all_results.append(result)

    write_run_summary(args=args, results=all_results, output_path=run_root / "run_summary.json")
    print("=" * 72)
    print("Classic baseline training finished / 经典 baseline 训练完成")
    print(f"Run root / 输出目录: {run_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
