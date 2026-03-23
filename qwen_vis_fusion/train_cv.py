from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen2VLProcessor

try:
    import wandb
except Exception:
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.path_utils import expand_env_placeholders, load_json_config, resolve_path
from qwen_vis_fusion.dataset import (
    CLASS_NAMES_2,
    CLASS_NAMES_3,
    QwenVisFusionDataset,
    available_input_modes,
    collate_fn,
    resolve_view_keys,
)
from qwen_vis_fusion.metrics import compute_classification_metrics
from qwen_vis_fusion.model import QwenVisionFusionClassifier


METHOD_CHOICES = ["baseline", "causal"]


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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    visual_params = []
    head_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("visual."):
            visual_params.append(param)
        elif name.startswith("fusion_head."):
            head_params.append(param)
        else:
            other_params.append(param)

    parameter_groups = []
    if visual_params:
        parameter_groups.append(
            {
                "params": visual_params,
                "lr": args.visual_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    if head_params:
        parameter_groups.append(
            {
                "params": head_params,
                "lr": args.fusion_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    if other_params:
        parameter_groups.append(
            {
                "params": other_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        )

    return torch.optim.AdamW(parameter_groups)


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.num_epochs))
    return None


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {
        **batch,
        "label_ids": batch["label_ids"].to(device),
    }
    if "pixel_values" in batch:
        moved["pixel_values"] = batch["pixel_values"].to(device)
    if "image_grid_thw" in batch:
        moved["image_grid_thw"] = batch["image_grid_thw"].to(device)
    if "image_vectors" in batch:
        moved["image_vectors"] = batch["image_vectors"].to(device)
    return moved


def forward_model(model: QwenVisionFusionClassifier, batch: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if "image_vectors" in batch:
        return model.forward_from_image_vectors(batch["image_vectors"])
    return model(
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
        sample_image_counts=batch["sample_image_counts"],
    )


@torch.no_grad()
def evaluate_model(
    model: QwenVisionFusionClassifier,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
    task_mode: str,
    subset_name: str,
) -> Dict[str, Any]:
    model.eval()
    y_true: List[str] = []
    y_pred: List[str] = []
    y_prob: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        logits, _ = forward_model(model, batch)
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
                    "condition_uid": batch["condition_uids"][idx],
                    "role": batch["roles"][idx],
                }
            )

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        y_prob=np.stack(y_prob, axis=0) if y_prob else None,
        subset_name=subset_name,
    )
    metrics["records"] = records
    return metrics


class CachedFusionDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def cached_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "image_vectors": torch.stack([item["image_vectors"] for item in batch], dim=0),
        "label_ids": torch.tensor([item["label_id"] for item in batch], dtype=torch.long),
        "label_names": [item["label_name"] for item in batch],
        "labels_2c": [item["label_2c"] for item in batch],
        "labels_3c": [item["label_3c"] for item in batch],
        "condition_uids": [item["condition_uid"] for item in batch],
        "roles": [item["role"] for item in batch],
        "positive_indices": [item.get("positive_index") for item in batch],
    }


class Trainer:
    def __init__(
        self,
        args: argparse.Namespace,
        model: QwenVisionFusionClassifier,
        train_dataset: QwenVisFusionDataset,
        val_dataset: QwenVisFusionDataset,
        test_dataset: QwenVisFusionDataset,
        run_dir: Path,
    ) -> None:
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.run_dir = run_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_names = class_names_for_mode(args.task_mode)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.optimizer = build_optimizer(args, model)
        self.scheduler = build_scheduler(args, self.optimizer)
        trainable_dtypes = {param.dtype for param in self.model.parameters() if param.requires_grad}
        self.use_grad_scaler = (
            self.device.type == "cuda"
            and args.amp
            and trainable_dtypes.issubset({torch.float32})
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        self.selection_metric = selection_metric_name(args.task_mode)
        self.use_cached_features = not self.model.has_trainable_visual
        self.cached_train_items: List[Dict[str, Any]] = []
        self.wandb_run = None

        if self.use_cached_features:
            self.cached_train_items = self._build_feature_cache(self.train_dataset)
            cached_val_items = self._build_feature_cache(self.val_dataset)
            cached_test_items = self._build_feature_cache(self.test_dataset)

            self.train_loader = DataLoader(
                CachedFusionDataset(self.cached_train_items),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=cached_collate_fn,
            )
            self.val_loader = DataLoader(
                CachedFusionDataset(cached_val_items),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=cached_collate_fn,
            )
            self.test_loader = DataLoader(
                CachedFusionDataset(cached_test_items),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=cached_collate_fn,
            )
            self.model.visual.cpu()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )

        self._init_wandb()

    def _init_wandb(self) -> None:
        if not self.args.use_wandb or wandb is None:
            return
        try:
            self.wandb_run = wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity or None,
                name=f"{self.args.wandb_run_prefix}-{self.run_dir.name}",
                reinit=True,
                config={
                    "experiment_name": self.args.experiment_name,
                    "task_mode": self.args.task_mode,
                    "method": self.args.method,
                    "input_mode": self.args.input_mode,
                    "scope": self.args.scope,
                    "fold": self.run_dir.name,
                    "learning_rate": self.args.learning_rate,
                    "visual_learning_rate": self.args.visual_learning_rate,
                    "fusion_learning_rate": self.args.fusion_learning_rate,
                    "weight_decay": self.args.weight_decay,
                    "batch_size": self.args.batch_size,
                    "hidden_dim": self.args.hidden_dim,
                    "unfreeze_last_n_blocks": self.args.unfreeze_last_n_blocks,
                    "causal_loss_weight": self.args.causal_loss_weight,
                    "init_run_root": self.args.init_run_root,
                },
            )
        except Exception:
            self.wandb_run = None

    def _build_feature_cache(self, dataset: QwenVisFusionDataset) -> List[Dict[str, Any]]:
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        cached_items: List[Dict[str, Any]] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, self.device)
                image_vectors = self.model.extract_image_vectors(
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    sample_image_counts=batch["sample_image_counts"],
                ).detach().cpu()
                for idx in range(image_vectors.shape[0]):
                    cached_items.append(
                        {
                            "sample_id": batch["sample_ids"][idx],
                            "image_vectors": image_vectors[idx],
                            "label_id": int(batch["label_ids"][idx].detach().cpu().item()),
                            "label_name": batch["label_names"][idx],
                            "label_2c": batch["labels_2c"][idx],
                            "label_3c": batch["labels_3c"][idx],
                            "condition_uid": batch["condition_uids"][idx],
                            "role": batch["roles"][idx],
                            "positive_index": batch["positive_indices"][idx],
                        }
                    )
        return cached_items

    def _compute_causal_loss(
        self,
        anchor_repr: torch.Tensor,
        positive_indices: List[Optional[int]],
    ) -> torch.Tensor:
        valid_pairs = [
            (anchor_idx, pos_idx)
            for anchor_idx, pos_idx in enumerate(positive_indices)
            if pos_idx is not None
        ]
        if not valid_pairs:
            return torch.zeros((), device=self.device, dtype=anchor_repr.dtype)

        with torch.no_grad():
            if self.use_cached_features:
                positive_vectors = torch.stack(
                    [self.cached_train_items[pos_idx]["image_vectors"] for _, pos_idx in valid_pairs],
                    dim=0,
                ).to(self.device)
                _, positive_repr = self.model.forward_from_image_vectors(positive_vectors)
            else:
                positive_items = [self.train_dataset[pos_idx] for _, pos_idx in valid_pairs]
                positive_batch = collate_fn(positive_items)
                positive_batch = move_batch_to_device(positive_batch, self.device)
                _, positive_repr = self.model(
                    pixel_values=positive_batch["pixel_values"],
                    image_grid_thw=positive_batch["image_grid_thw"],
                    sample_image_counts=positive_batch["sample_image_counts"],
                )

        anchor_positions = torch.tensor(
            [anchor_idx for anchor_idx, _ in valid_pairs],
            device=self.device,
            dtype=torch.long,
        )
        matched_anchor_repr = anchor_repr.index_select(0, anchor_positions)
        cosine_sim = torch.nn.functional.cosine_similarity(matched_anchor_repr, positive_repr, dim=-1)
        return (1.0 - cosine_sim).mean()

    def _write_eval_outputs(self, stage: str, split: str, metrics: Dict[str, Any]) -> None:
        split_dir = self.run_dir / "evals" / stage / split
        split_dir.mkdir(parents=True, exist_ok=True)
        records = metrics.pop("records", [])
        save_json(split_dir / "metrics_summary.json", metrics)
        if records:
            with (split_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        if stage == "after":
            legacy_dir = self.run_dir / "evals" / split
            legacy_dir.mkdir(parents=True, exist_ok=True)
            save_json(legacy_dir / "metrics_summary.json", metrics)
            if records:
                with (legacy_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)

    def train(self) -> Dict[str, Any]:
        best_metric = -float("inf")
        best_epoch = 0
        best_state: Optional[Dict[str, Any]] = None
        patience_counter = 0
        history: List[Dict[str, Any]] = []

        for epoch in range(1, self.args.num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_ce = 0.0
            total_causal = 0.0

            for batch in self.train_loader:
                batch = move_batch_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)

                autocast_enabled = self.device.type == "cuda" and self.args.amp
                with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=torch.float16):
                    logits, fused_repr = forward_model(self.model, batch)
                    ce_loss = self.criterion(logits, batch["label_ids"])
                    causal_loss = torch.zeros((), device=self.device, dtype=ce_loss.dtype)
                    if self.args.method == "causal" and self.args.causal_loss_weight > 0:
                        causal_loss = self._compute_causal_loss(
                            anchor_repr=fused_repr,
                            positive_indices=batch["positive_indices"],
                        )
                    loss = ce_loss + self.args.causal_loss_weight * causal_loss

                if self.use_grad_scaler:
                    self.scaler.scale(loss).backward()
                    if self.args.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                total_loss += float(loss.detach().item())
                total_ce += float(ce_loss.detach().item())
                total_causal += float(causal_loss.detach().item())

            if self.scheduler is not None:
                self.scheduler.step()

            val_metrics = evaluate_model(
                model=self.model,
                dataloader=self.val_loader,
                class_names=self.class_names,
                device=self.device,
                task_mode=self.args.task_mode,
                subset_name=f"{self.args.scope}_{self.run_dir.name}_val",
            )
            current_metric = float(val_metrics.get(self.selection_metric, 0.0))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(1, len(self.train_loader)),
                    "train_ce_loss": total_ce / max(1, len(self.train_loader)),
                    "train_causal_loss": total_causal / max(1, len(self.train_loader)),
                    "val_metrics": val_metrics,
                }
            )
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": total_loss / max(1, len(self.train_loader)),
                        "train/ce_loss": total_ce / max(1, len(self.train_loader)),
                        "train/causal_loss": total_causal / max(1, len(self.train_loader)),
                        "val/accuracy": val_metrics["accuracy"],
                        "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                        "val/macro_f1": val_metrics["macro_f1"],
                    }
                )

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_state = {
                    "model": self.model.state_dict(),
                    "epoch": epoch,
                    "metric": current_metric,
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.args.early_stopping_patience:
                    break

        assert best_state is not None
        torch.save(best_state, self.run_dir / "best.pt")
        self.model.load_state_dict(best_state["model"])

        val_metrics = evaluate_model(
            model=self.model,
            dataloader=self.val_loader,
            class_names=self.class_names,
            device=self.device,
            task_mode=self.args.task_mode,
            subset_name=f"{self.args.scope}_{self.run_dir.name}_val",
        )
        test_metrics = evaluate_model(
            model=self.model,
            dataloader=self.test_loader,
            class_names=self.class_names,
            device=self.device,
            task_mode=self.args.task_mode,
            subset_name=f"{self.args.scope}_{self.run_dir.name}_test",
        )
        self._write_eval_outputs("after", "val", val_metrics.copy())
        self._write_eval_outputs("after", "test", test_metrics.copy())
        save_json(self.run_dir / "logs" / "train_log.json", {"history": history})
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "best_epoch": best_epoch,
                    "after/val_accuracy": val_metrics["accuracy"],
                    "after/val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "after/val_macro_f1": val_metrics["macro_f1"],
                    "after/test_accuracy": test_metrics["accuracy"],
                    "after/test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "after/test_macro_f1": test_metrics["macro_f1"],
                }
            )
            self.wandb_run.finish()

        return {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "selection_metric_name": self.selection_metric,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }


def train_one_fold(
    args: argparse.Namespace,
    protocol_root: Path,
    processor: Qwen2VLProcessor,
    fold_name: str,
) -> Dict[str, Any]:
    run_dir = fold_run_dir(Path(args.run_root), args.scope, fold_name)
    for subdir in ["evals", "logs"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    train_dataset = QwenVisFusionDataset(
        manifest_path=manifest_path(protocol_root, args.scope, fold_name, "train"),
        processor=processor,
        task_mode=args.task_mode,
        input_mode=args.input_mode,
        positive_seed=args.seed,
        random_positive=True,
    )
    val_dataset = QwenVisFusionDataset(
        manifest_path=manifest_path(protocol_root, args.scope, fold_name, "val"),
        processor=processor,
        task_mode=args.task_mode,
        input_mode=args.input_mode,
        positive_seed=args.seed,
        random_positive=False,
    )
    test_dataset = QwenVisFusionDataset(
        manifest_path=manifest_path(protocol_root, args.scope, fold_name, "test"),
        processor=processor,
        task_mode=args.task_mode,
        input_mode=args.input_mode,
        positive_seed=args.seed,
        random_positive=False,
    )

    model = QwenVisionFusionClassifier(
        model_name_or_path=args.model_name_or_path,
        num_views=len(resolve_view_keys(args.input_mode)),
        num_classes=len(class_names_for_mode(args.task_mode)),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        model_dtype=args.model_dtype,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
    )

    init_checkpoint_path: Optional[Path] = None
    before_val_metrics: Optional[Dict[str, Any]] = None
    before_test_metrics: Optional[Dict[str, Any]] = None
    if args.init_run_root:
        init_checkpoint_path = fold_run_dir(Path(args.init_run_root), args.scope, fold_name) / "best.pt"
        checkpoint = torch.load(init_checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        run_dir=run_dir,
    )

    if args.eval_before_training and init_checkpoint_path is not None:
        before_val_metrics = evaluate_model(
            model=trainer.model,
            dataloader=trainer.val_loader,
            class_names=trainer.class_names,
            device=trainer.device,
            task_mode=args.task_mode,
            subset_name=f"{args.scope}_{run_dir.name}_before_val",
        )
        before_test_metrics = evaluate_model(
            model=trainer.model,
            dataloader=trainer.test_loader,
            class_names=trainer.class_names,
            device=trainer.device,
            task_mode=args.task_mode,
            subset_name=f"{args.scope}_{run_dir.name}_before_test",
        )
        trainer._write_eval_outputs("before", "val", before_val_metrics.copy())
        trainer._write_eval_outputs("before", "test", before_test_metrics.copy())
        if trainer.wandb_run is not None:
            trainer.wandb_run.log(
                {
                    "before/val_accuracy": before_val_metrics["accuracy"],
                    "before/val_balanced_accuracy": before_val_metrics["balanced_accuracy"],
                    "before/val_macro_f1": before_val_metrics["macro_f1"],
                    "before/test_accuracy": before_test_metrics["accuracy"],
                    "before/test_balanced_accuracy": before_test_metrics["balanced_accuracy"],
                    "before/test_macro_f1": before_test_metrics["macro_f1"],
                }
            )

    result = trainer.train()

    save_json(
        run_dir / "config.json",
        {
            "experiment_name": args.experiment_name,
            "task_mode": args.task_mode,
            "method": args.method,
            "input_mode": args.input_mode,
            "num_views": len(resolve_view_keys(args.input_mode)),
            "model_name_or_path": args.model_name_or_path,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "visual_learning_rate": args.visual_learning_rate,
            "fusion_learning_rate": args.fusion_learning_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "causal_loss_weight": args.causal_loss_weight,
            "max_grad_norm": args.max_grad_norm,
            "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
            "init_run_root": args.init_run_root,
            "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path is not None else None,
            "train_manifest": str(manifest_path(protocol_root, args.scope, fold_name, "train")),
            "val_manifest": str(manifest_path(protocol_root, args.scope, fold_name, "val")),
            "test_manifest": str(manifest_path(protocol_root, args.scope, fold_name, "test")),
        },
    )

    return {
        "fold": fold_name,
        "before_val_metrics": before_val_metrics,
        "before_test_metrics": before_test_metrics,
        "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path is not None else None,
        **result,
    }


def write_run_summary(args: argparse.Namespace, results: List[Dict[str, Any]], output_path: Path) -> None:
    payload = {
        "experiment_name": args.experiment_name,
        "task_mode": args.task_mode,
        "method": args.method,
        "input_mode": args.input_mode,
        "model_name_or_path": args.model_name_or_path,
        "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
        "init_run_root": args.init_run_root,
        "run_root": args.run_root,
        "fold_results": results,
    }
    save_json(output_path, payload)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    known, _ = config_parser.parse_known_args()
    config_defaults = load_json(known.config) if known.config else {}

    def cfg(name: str, fallback):
        return config_defaults.get(name, fallback)

    parser = argparse.ArgumentParser(description="Train Qwen vision fusion baselines on the current data1 protocol.")
    parser.add_argument("--config", type=str, default=known.config)
    parser.add_argument("--protocol_root", type=str, default=cfg("protocol_root", "${SLM_DATA1_PROTOCOL_ROOT}"))
    parser.add_argument("--scope", type=str, default=cfg("scope", "official_cv"), choices=["official_cv", "quick_dev"])
    parser.add_argument("--task_mode", type=str, default=cfg("task_mode", None), choices=["2class", "3class"])
    parser.add_argument("--method", type=str, default=cfg("method", "baseline"), choices=METHOD_CHOICES)
    parser.add_argument("--input_mode", type=str, default=cfg("input_mode", "rgb_dual"), choices=available_input_modes())
    parser.add_argument("--experiment_name", type=str, default=cfg("experiment_name", None))
    parser.add_argument("--model_name_or_path", type=str, default=cfg("model_name_or_path", "${QWEN2_VL_MODEL_PATH}"))
    parser.add_argument("--run_root", type=str, default=cfg("run_root", None))
    parser.add_argument("--fold", type=str, default=cfg("fold", "all"))
    parser.add_argument("--batch_size", type=int, default=cfg("batch_size", 4))
    parser.add_argument("--num_epochs", type=int, default=cfg("num_epochs", 8))
    parser.add_argument("--early_stopping_patience", type=int, default=cfg("early_stopping_patience", 2))
    parser.add_argument("--learning_rate", type=float, default=cfg("learning_rate", 3e-4))
    parser.add_argument("--weight_decay", type=float, default=cfg("weight_decay", 1e-4))
    parser.add_argument("--visual_learning_rate", type=float, default=cfg("visual_learning_rate", 1e-5))
    parser.add_argument("--fusion_learning_rate", type=float, default=cfg("fusion_learning_rate", 3e-4))
    parser.add_argument("--dropout", type=float, default=cfg("dropout", 0.1))
    parser.add_argument("--hidden_dim", type=int, default=cfg("hidden_dim", 512))
    parser.add_argument("--num_workers", type=int, default=cfg("num_workers", 0))
    parser.add_argument("--seed", type=int, default=cfg("seed", 42))
    parser.add_argument("--scheduler", type=str, default=cfg("scheduler", "cosine"), choices=["none", "cosine"])
    parser.add_argument("--label_smoothing", type=float, default=cfg("label_smoothing", 0.0))
    parser.add_argument("--amp", type=str2bool, default=cfg("amp", True))
    parser.add_argument("--model_dtype", type=str, default=cfg("model_dtype", "float16"), choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=cfg("unfreeze_last_n_blocks", 0))
    parser.add_argument("--causal_loss_weight", type=float, default=cfg("causal_loss_weight", 0.0))
    parser.add_argument("--max_grad_norm", type=float, default=cfg("max_grad_norm", 1.0))
    parser.add_argument("--init_run_root", type=str, default=cfg("init_run_root", None))
    parser.add_argument("--eval_before_training", type=str2bool, default=cfg("eval_before_training", False))
    parser.add_argument("--use_wandb", type=str2bool, default=cfg("use_wandb", False))
    parser.add_argument("--wandb_project", type=str, default=cfg("wandb_project", "qwen_vis_fusion"))
    parser.add_argument("--wandb_entity", type=str, default=cfg("wandb_entity", ""))
    parser.add_argument("--wandb_run_prefix", type=str, default=cfg("wandb_run_prefix", "qwen-vis"))
    args = parser.parse_args()

    if not args.task_mode:
        raise ValueError("--task_mode is required")
    if not args.run_root:
        raise ValueError("--run_root is required")
    if args.experiment_name is None:
        args.experiment_name = f"qwen_vis_{args.input_mode}_{args.method}_{args.task_mode}"
    args.protocol_root = expand_env_placeholders(args.protocol_root)
    args.model_name_or_path = expand_env_placeholders(args.model_name_or_path)
    args.run_root = expand_env_placeholders(args.run_root)
    if args.init_run_root:
        args.init_run_root = expand_env_placeholders(args.init_run_root)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    protocol_root = resolve_path(args.protocol_root)
    run_root = resolve_path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    processor = Qwen2VLProcessor.from_pretrained(args.model_name_or_path, use_fast=False)
    folds = available_folds(protocol_root, args.scope)
    if args.fold != "all":
        folds = [args.fold]

    all_results = []
    for fold in folds:
        print("=" * 72)
        print(
            "Training qwen vision fusion: "
            f"exp={args.experiment_name}, task={args.task_mode}, "
            f"method={args.method}, input={args.input_mode}, fold={fold}"
        )
        print("=" * 72)
        result = train_one_fold(args=args, protocol_root=protocol_root, processor=processor, fold_name=fold)
        all_results.append(result)

    write_run_summary(args=args, results=all_results, output_path=run_root / "run_summary.json")
    print("=" * 72)
    print("Qwen vision fusion training finished / Qwen 视觉融合训练完成")
    print(f"Run root / 输出目录: {run_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
