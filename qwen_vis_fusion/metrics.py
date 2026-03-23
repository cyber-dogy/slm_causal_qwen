from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


CLASS_NAMES_2 = ["normal", "abnormal"]
CLASS_NAMES_3 = ["normal", "HEW", "LEL"]


def fold_3class_to_2class(label: str) -> str:
    return "normal" if label == "normal" else "abnormal"


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    class_names: List[str],
    y_prob: Optional[np.ndarray] = None,
    subset_name: str = "eval",
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "subset_name": subset_name,
        "class_names": class_names,
        "n_samples": len(y_true),
        "unknown_rate": 0.0,
    }

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0))
    metrics["macro_precision"] = float(
        precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, average="macro", zero_division=0
        )[0]
    )
    metrics["macro_recall"] = float(
        precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, average="macro", zero_division=0
        )[1]
    )
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_names,
        average=None,
        zero_division=0,
    )
    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=class_names).tolist()

    if y_prob is not None and len(class_names) >= 2:
        try:
            if len(class_names) == 2:
                metrics["roc_auc"] = {
                    "macro": float(roc_auc_score(y_true, y_prob[:, 1], labels=class_names)),
                }
            else:
                metrics["roc_auc"] = {
                    "macro": float(
                        roc_auc_score(
                            y_true,
                            y_prob,
                            labels=class_names,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
                }
        except Exception:
            pass

    if class_names == CLASS_NAMES_3:
        y_true_2c = [fold_3class_to_2class(label) for label in y_true]
        y_pred_2c = [fold_3class_to_2class(label) for label in y_pred]
        metrics["folded_2class"] = compute_classification_metrics(
            y_true=y_true_2c,
            y_pred=y_pred_2c,
            class_names=CLASS_NAMES_2,
            y_prob=None,
            subset_name=f"{subset_name}_folded_2class",
        )

    return metrics

