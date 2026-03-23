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
                positive_index = 1
                metrics["roc_auc"] = {
                    "macro": float(roc_auc_score(y_true, y_prob[:, positive_index], labels=class_names)),
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

    return metrics
