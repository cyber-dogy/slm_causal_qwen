"""
Qwen2-VL SLM Defect Detection - Metrics Utilities

This module provides comprehensive metric computation for classification evaluation,
including standard classification metrics, confusion matrix visualization, and
ROC/PR curve analysis.
"""

import json
import csv
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
)


# ============================================================================
# Constants
# ============================================================================

# 为了向后兼容保留旧常量
CLASS_NAMES = ["normal", "HEW", "LEL"]
CLASS_NAME_TO_ID = {"normal": 0, "HEW": 1, "LEL": 2}
ID_TO_CLASS_NAME = {0: "normal", 1: "HEW", 2: "LEL"}

# 新增二分类常量
CLASS_NAMES_2 = ["normal", "abnormal"]
CLASS_NAME_TO_ID_2 = {"normal": 0, "abnormal": 1}
ID_TO_CLASS_NAME_2 = {0: "normal", 1: "abnormal"}


# ============================================================================
# Prediction Normalization
# ============================================================================

def normalize_prediction(text: str) -> str:
    """
    Normalize model-generated text to one of the three class names.
    
    Args:
        text: Raw text output from the model
        
    Returns:
        One of: "normal", "HEW", "LEL", or "UNKNOWN"
    """
    if not text:
        return "UNKNOWN"
    
    # Clean and upper case for matching
    text_clean = text.strip().lower()
    
    # Direct match patterns
    if "hew" in text_clean:
        # Check for false positives like "unhew" or "show"
        # Require hew to be a standalone word or at word boundary
        import re
        if re.search(r'\bhew\b', text_clean):
            return "HEW"
    
    if "lel" in text_clean:
        import re
        if re.search(r'\blel\b', text_clean):
            return "LEL"
    
    if "normal" in text_clean:
        import re
        if re.search(r'\bnormal\b', text_clean):
            return "normal"
    
    # Check for exact matches (case insensitive)
    text_upper = text.strip().upper()
    if text_upper == "HEW":
        return "HEW"
    if text_upper == "LEL":
        return "LEL"
    if text_upper == "NORMAL":
        return "normal"
    
    # Check first word
    first_word = text_clean.split()[0] if text_clean.split() else ""
    if first_word in ["hew"]:
        return "HEW"
    if first_word in ["lel"]:
        return "LEL"
    if first_word in ["normal"]:
        return "normal"
    
    # Check last word
    last_word = text_clean.split()[-1] if text_clean.split() else ""
    if last_word.rstrip('.!?,:;') in ["hew"]:
        return "HEW"
    if last_word.rstrip('.!?,:;') in ["lel"]:
        return "LEL"
    if last_word.rstrip('.!?,:;') in ["normal"]:
        return "normal"
    
    return "UNKNOWN"


def normalize_predictions(predictions: List[str]) -> List[str]:
    """Normalize a list of predictions."""
    return [normalize_prediction(p) for p in predictions]


# ============================================================================
# Metric Computation
# ============================================================================

def compute_classification_metrics(
    y_true: List[Union[str, int]],
    y_pred: List[Union[str, int]],
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    subset_name: str = "default",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (strings or integers)
        y_pred: Predicted labels (strings or integers)
        y_probs: Predicted probabilities for each class [n_samples, n_classes]
        class_names: List of class names (auto-detected if None)
        subset_name: Name of the subset being evaluated
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Auto-detect class names if not provided
    if class_names is None:
        unique_labels = set(str(y) for y in set(y_true) | set(y_pred) if str(y) != "UNKNOWN")
        if unique_labels <= {"normal", "HEW", "LEL"}:
            class_names = CLASS_NAMES
        elif unique_labels <= {"normal", "abnormal"}:
            class_names = CLASS_NAMES_2
        else:
            class_names = sorted(unique_labels)
    label_to_id = {name: idx for idx, name in enumerate(class_names)}

    def to_label_id(value: Union[str, int]) -> int:
        if isinstance(value, (int, np.integer)):
            return int(value)

        value_str = str(value)
        if value_str == "UNKNOWN":
            return -1

        return label_to_id.get(value_str, -1)

    y_true_ids = np.array([to_label_id(y) for y in y_true], dtype=np.int64)
    y_pred_ids = np.array([to_label_id(y) for y in y_pred], dtype=np.int64)
    
    n_classes = len(class_names)
    
    # Check for unknown predictions
    unknown_mask = y_pred_ids < 0
    invalid_true_mask = y_true_ids < 0
    
    unknown_rate = unknown_mask.mean()
    valid_mask = (~unknown_mask) & (~invalid_true_mask)
    
    # Basic counts
    n_total = len(y_true_ids)
    n_valid = valid_mask.sum()
    n_correct = (y_true_ids[valid_mask] == y_pred_ids[valid_mask]).sum()
    
    # Standard metrics (only on valid predictions)
    if n_valid > 0:
        accuracy = accuracy_score(y_true_ids[valid_mask], y_pred_ids[valid_mask])
        balanced_acc = balanced_accuracy_score(y_true_ids[valid_mask], y_pred_ids[valid_mask])
        
        # Per-class metrics
        precision_per_class = precision_score(
            y_true_ids[valid_mask], y_pred_ids[valid_mask], 
            average=None, labels=range(n_classes), zero_division=0
        )
        recall_per_class = recall_score(
            y_true_ids[valid_mask], y_pred_ids[valid_mask], 
            average=None, labels=range(n_classes), zero_division=0
        )
        f1_per_class = f1_score(
            y_true_ids[valid_mask], y_pred_ids[valid_mask], 
            average=None, labels=range(n_classes), zero_division=0
        )
        support_per_class = np.bincount(y_true_ids[valid_mask], minlength=n_classes)
        
        # Macro and weighted averages
        macro_precision = precision_score(
            y_true_ids[valid_mask], y_pred_ids[valid_mask], 
            average="macro", zero_division=0
        )
        macro_recall = recall_score(
            y_true_ids[valid_mask], y_pred_ids[valid_mask], 
            average="macro", zero_division=0
        )
        macro_f1 = f1_score(
            y_true_ids[valid_mask], y_pred_ids[valid_mask], 
            average="macro", zero_division=0
        )
        
        weighted_precision = precision_score(
            y_true_ids[valid_mask], y_pred_ids[valid_mask], 
            average="weighted", zero_division=0
        )
        weighted_recall = recall_score(
            y_true_ids[valid_mask], y_pred_ids[valid_mask], 
            average="weighted", zero_division=0
        )
        weighted_f1 = f1_score(
            y_true_ids[valid_mask], y_pred_ids[valid_mask], 
            average="weighted", zero_division=0
        )
        
        # MCC and Cohen's Kappa
        try:
            mcc = matthews_corrcoef(y_true_ids[valid_mask], y_pred_ids[valid_mask])
        except Exception:
            mcc = None
            
        try:
            cohen_kappa = cohen_kappa_score(y_true_ids[valid_mask], y_pred_ids[valid_mask])
        except Exception:
            cohen_kappa = None
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_ids[valid_mask], y_pred_ids[valid_mask], labels=range(n_classes))
    else:
        # No valid predictions
        accuracy = 0.0
        balanced_acc = 0.0
        precision_per_class = np.zeros(n_classes)
        recall_per_class = np.zeros(n_classes)
        f1_per_class = np.zeros(n_classes)
        support_per_class = np.zeros(n_classes)
        macro_precision = macro_recall = macro_f1 = 0.0
        weighted_precision = weighted_recall = weighted_f1 = 0.0
        mcc = cohen_kappa = None
        cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # ROC-AUC and PR-AUC (if probabilities available)
    roc_auc_metrics = {}
    pr_auc_metrics = {}
    
    if y_probs is not None and len(y_probs) > 0:
        # Filter valid samples
        y_probs_valid = y_probs[valid_mask] if len(y_probs) == len(y_true_ids) else y_probs
        y_true_valid = y_true_ids[valid_mask]
        
        if len(y_probs_valid) > 0:
            # One-vs-Rest ROC-AUC
            try:
                # Binarize labels
                y_true_bin = np.zeros((len(y_true_valid), n_classes))
                for i, y in enumerate(y_true_valid):
                    if 0 <= y < n_classes:
                        y_true_bin[i, y] = 1
                
                # Per-class ROC-AUC
                per_class_roc_auc = {}
                for i, class_name in enumerate(class_names):
                    if y_true_bin[:, i].sum() > 0 and y_true_bin[:, i].sum() < len(y_true_bin):
                        try:
                            auc = roc_auc_score(y_true_bin[:, i], y_probs_valid[:, i])
                            per_class_roc_auc[class_name] = float(auc)
                        except Exception:
                            pass
                
                # Macro ROC-AUC
                if per_class_roc_auc:
                    roc_auc_metrics["macro"] = np.mean(list(per_class_roc_auc.values()))
                    roc_auc_metrics["per_class"] = per_class_roc_auc
                
                # Micro ROC-AUC (less common for multi-class)
                try:
                    roc_auc_metrics["micro"] = roc_auc_score(
                        y_true_bin.ravel(), y_probs_valid.ravel()
                    )
                except Exception:
                    pass
                    
            except Exception as e:
                warnings.warn(f"Could not compute ROC-AUC: {e}")
            
            # PR-AUC
            try:
                per_class_pr_auc = {}
                for i, class_name in enumerate(class_names):
                    if y_true_bin[:, i].sum() > 0:
                        try:
                            auc = average_precision_score(y_true_bin[:, i], y_probs_valid[:, i])
                            per_class_pr_auc[class_name] = float(auc)
                        except Exception:
                            pass
                
                if per_class_pr_auc:
                    pr_auc_metrics["macro"] = np.mean(list(per_class_pr_auc.values()))
                    pr_auc_metrics["per_class"] = per_class_pr_auc
                    
            except Exception as e:
                warnings.warn(f"Could not compute PR-AUC: {e}")
    
    # Build results dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support_per_class[i]),
        }
    
    results = {
        "subset_name": subset_name,
        "class_names": list(class_names),
        "n_samples": n_total,
        "n_valid_predictions": int(n_valid),
        "unknown_rate": float(unknown_rate),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "mcc": float(mcc) if mcc is not None else None,
        "cohen_kappa": float(cohen_kappa) if cohen_kappa is not None else None,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }
    
    if roc_auc_metrics:
        results["roc_auc"] = roc_auc_metrics
    if pr_auc_metrics:
        results["pr_auc"] = pr_auc_metrics
    
    return results


def compute_transfer_gap(val_metrics: Dict, test_metrics: Dict) -> Dict[str, float]:
    """
    Compute transfer gap between validation and test metrics.
    
    Args:
        val_metrics: Metrics dictionary from validation set
        test_metrics: Metrics dictionary from test set
        
    Returns:
        Dictionary of gap metrics
    """
    gaps = {}
    
    metrics_to_compare = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]
    
    for metric in metrics_to_compare:
        val_val = val_metrics.get(metric, 0)
        test_val = test_metrics.get(metric, 0)
        if val_val is not None and test_val is not None:
            gaps[f"transfer_gap_{metric}"] = val_val - test_val
    
    # Per-class recall gaps
    val_per_class = val_metrics.get("per_class", {})
    test_per_class = test_metrics.get("per_class", {})
    
    class_names = val_metrics.get("class_names") or test_metrics.get("class_names") or CLASS_NAMES

    for class_name in class_names:
        val_recall = val_per_class.get(class_name, {}).get("recall", 0)
        test_recall = test_per_class.get(class_name, {}).get("recall", 0)
        gaps[f"transfer_gap_recall_{class_name}"] = val_recall - test_recall
    
    return gaps


# ============================================================================
# Visualization
# ============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize for percentages (optional - show both)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        # Plot heatmap
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Proportion'},
        )
        
        # Add raw counts as text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.texts[i * len(class_names) + j]
                count = int(cm[i, j])
                percentage = float(cm_norm[i, j])
                text.set_text(f'{count}\n({percentage:.1%})')
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        warnings.warn("matplotlib/seaborn not available, skipping confusion matrix plot")
    except Exception as e:
        warnings.warn(f"Failed to plot confusion matrix: {e}")


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict],
    output_path: str,
    title: str = "Metrics Comparison",
):
    """
    Plot comparison of metrics across multiple runs/subsets.
    
    Args:
        metrics_dict: Dict mapping names to metrics dictionaries
        output_path: Path to save the figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        metrics_to_plot = [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(metrics_dict)
        
        for i, (name, metrics) in enumerate(metrics_dict.items()):
            values = [metrics.get(m, 0) for m in metrics_to_plot]
            ax.bar(x + i * width, values, width, label=name)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(metrics_dict) - 1) / 2)
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics_to_plot])
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        warnings.warn("matplotlib not available, skipping metrics comparison plot")
    except Exception as e:
        warnings.warn(f"Failed to plot metrics comparison: {e}")


# ============================================================================
# Saving/Loading Results
# ============================================================================

def save_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
    prefix: str = "metrics",
):
    """
    Save metrics to multiple formats.
    
    Args:
        metrics: Metrics dictionary
        output_dir: Directory to save files
        prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{prefix}_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Save as CSV (summary)
    csv_path = os.path.join(output_dir, f"{prefix}_summary.csv")
    summary_data = {
        k: v for k, v in metrics.items()
        if k not in ["per_class", "confusion_matrix", "roc_auc", "pr_auc"]
    }
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        writer.writeheader()
        writer.writerow(summary_data)
    
    # Save per-class metrics
    per_class_path = os.path.join(output_dir, f"{prefix}_per_class.csv")
    per_class = metrics.get("per_class", {})
    class_names = metrics.get("class_names", CLASS_NAMES)
    if per_class:
        with open(per_class_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["class", "precision", "recall", "f1", "support"])
            for class_name, class_metrics in per_class.items():
                writer.writerow([
                    class_name,
                    class_metrics.get("precision", 0),
                    class_metrics.get("recall", 0),
                    class_metrics.get("f1", 0),
                    class_metrics.get("support", 0),
                ])
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.csv")
    cm = metrics.get("confusion_matrix", [])
    if cm:
        with open(cm_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([""] + class_names)
            for i, row in enumerate(cm):
                row_label = class_names[i] if i < len(class_names) else f"class_{i}"
                writer.writerow([row_label] + row)
    
    # Plot confusion matrix
    if cm:
        plot_confusion_matrix(
            np.array(cm),
            class_names,
            os.path.join(output_dir, f"{prefix}_confusion_matrix.png"),
            title=f"Confusion Matrix - {metrics.get('subset_name', 'Unknown')}",
        )


def load_predictions(predictions_path: str) -> List[Dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def save_predictions(
    predictions: List[Dict],
    output_path: str,
):
    """
    Save predictions to JSONL file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    # Also save errors separately
    errors = [p for p in predictions if not p.get("is_correct", False)]
    if errors:
        error_path = output_path.replace(".jsonl", "_errors.csv")
        with open(error_path, 'w', newline='', encoding='utf-8') as f:
            if errors:
                writer = csv.DictWriter(f, fieldnames=errors[0].keys())
                writer.writeheader()
                writer.writerows(errors)


def save_transfer_gap(
    gaps: Dict[str, float],
    output_dir: str,
):
    """Save transfer gap metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    json_path = os.path.join(output_dir, "transfer_gap.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(gaps, f, indent=2)
    
    # CSV
    csv_path = os.path.join(output_dir, "transfer_gap.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "gap_value"])
        for metric, value in gaps.items():
            writer.writerow([metric, value])


# ============================================================================
# Comparison Report Generation
# ============================================================================

def generate_before_after_comparison(
    before_metrics: Dict[str, Dict],
    after_metrics: Dict[str, Dict],
    output_path: str,
):
    """
    Generate a comprehensive before/after comparison report.
    
    Args:
        before_metrics: Dict mapping subset names to metrics (before fine-tuning)
        after_metrics: Dict mapping subset names to metrics (after fine-tuning)
        output_path: Path to save the markdown report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    lines = [
        "# Fine-Tuning Before/After Comparison Report\n",
        f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Summary\n",
    ]
    
    # Overall comparison table
    lines.append("### Overall Metrics Comparison\n")
    lines.append("| Metric | Before FT | After FT | Delta |\n")
    lines.append("|--------|-----------|----------|-------|\n")
    
    metrics_to_compare = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_recall",
    ]
    
    for subset in before_metrics.keys():
        if subset not in after_metrics:
            continue
            
        lines.append(f"\n#### Subset: {subset}\n")
        lines.append("| Metric | Before FT | After FT | Delta |\n")
        lines.append("|--------|-----------|----------|-------|\n")
        
        for metric in metrics_to_compare:
            before_val = before_metrics[subset].get(metric, 0)
            after_val = after_metrics[subset].get(metric, 0)
            delta = after_val - before_val
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            lines.append(f"| {metric} | {before_val:.4f} | {after_val:.4f} | {delta_str} |\n")
    
    # Per-class comparison
    lines.append("\n## Per-Class Metrics Comparison\n")
    
    for subset in before_metrics.keys():
        if subset not in after_metrics:
            continue
            
        lines.append(f"\n### Subset: {subset}\n")
        lines.append("| Class | Metric | Before FT | After FT | Delta |\n")
        lines.append("|-------|--------|-----------|----------|-------|\n")
        
        before_per_class = before_metrics[subset].get("per_class", {})
        after_per_class = after_metrics[subset].get("per_class", {})
        class_names = (
            before_metrics[subset].get("class_names")
            or after_metrics[subset].get("class_names")
            or list(dict.fromkeys(list(before_per_class.keys()) + list(after_per_class.keys())))
            or CLASS_NAMES
        )
        
        for class_name in class_names:
            before_class_metrics = before_per_class.get(class_name, {})
            after_class_metrics = after_per_class.get(class_name, {})
            
            for metric in ["precision", "recall", "f1"]:
                before_val = before_class_metrics.get(metric, 0)
                after_val = after_class_metrics.get(metric, 0)
                delta = after_val - before_val
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                lines.append(f"| {class_name} | {metric} | {before_val:.4f} | {after_val:.4f} | {delta_str} |\n")
    
    # Transfer gap analysis
    lines.append("\n## Transfer Gap Analysis\n")
    lines.append("Transfer gap = Validation - Test (higher indicates more overfitting to source)\n\n")
    
    for label, metrics_dict in [("Before FT", before_metrics), ("After FT", after_metrics)]:
        if "val" in metrics_dict and "test" in metrics_dict:
            val_metrics = metrics_dict["val"]
            test_metrics = metrics_dict["test"]
            
            lines.append(f"\n### {label}\n")
            lines.append("| Metric | Validation | Test | Gap |\n")
            lines.append("|--------|------------|------|-----|\n")
            
            for metric in ["balanced_accuracy", "macro_f1"]:
                val_val = val_metrics.get(metric, 0)
                test_val = test_metrics.get(metric, 0)
                gap = val_val - test_val
                gap_str = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
                lines.append(f"| {metric} | {val_val:.4f} | {test_val:.4f} | {gap_str} |\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Comparison report saved to: {output_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def print_metrics_table(metrics: Dict[str, Any], title: str = "Metrics"):
    """Pretty print metrics to console."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"\nSubset: {metrics.get('subset_name', 'Unknown')}")
    print(f"Total Samples: {metrics.get('n_samples', 0)}")
    print(f"Valid Predictions: {metrics.get('n_valid_predictions', 0)}")
    print(f"Unknown Rate: {metrics.get('unknown_rate', 0):.4f}")
    
    print(f"\n--- Overall Metrics ---")
    print(f"Accuracy:            {metrics.get('accuracy', 0):.4f}")
    print(f"Balanced Accuracy:   {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"Macro F1:            {metrics.get('macro_f1', 0):.4f}")
    print(f"Weighted F1:         {metrics.get('weighted_f1', 0):.4f}")
    
    if metrics.get('mcc') is not None:
        print(f"MCC:                 {metrics.get('mcc', 0):.4f}")
    if metrics.get('cohen_kappa') is not None:
        print(f"Cohen's Kappa:       {metrics.get('cohen_kappa', 0):.4f}")
    
    print(f"\n--- Per-Class Metrics ---")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 50)
    
    per_class = metrics.get("per_class", {})
    class_names = metrics.get("class_names") or list(per_class.keys()) or CLASS_NAMES
    for class_name in class_names:
        cm = per_class.get(class_name, {})
        print(f"{class_name:<10} {cm.get('precision', 0):<10.4f} {cm.get('recall', 0):<10.4f} {cm.get('f1', 0):<10.4f} {cm.get('support', 0):<10}")
    
    print(f"\n--- Confusion Matrix ---")
    cm = metrics.get("confusion_matrix", [])
    if cm:
        print(f"{'':<10}", end="")
        for name in class_names:
            print(f"{name:<10}", end="")
        print()
        for i, row in enumerate(cm):
            row_label = class_names[i] if i < len(class_names) else f"class_{i}"
            print(f"{row_label:<10}", end="")
            for val in row:
                print(f"{val:<10}", end="")
            print()
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the metrics computation
    y_true = ["normal", "HEW", "LEL", "normal", "HEW", "LEL", "normal", "normal"]
    y_pred = ["normal", "HEW", "HEW", "normal", "LEL", "LEL", "normal", "HEW"]
    
    metrics = compute_classification_metrics(y_true, y_pred, subset_name="test")
    print_metrics_table(metrics)
    
    # Test normalization
    test_cases = [
        "HEW",
        "hew",
        "The answer is HEW",
        "normal",
        "normal.",
        "lel",
        "LEL",
        "unknown",
        "",
    ]
    print("\nNormalization tests:")
    for tc in test_cases:
        print(f"  '{tc}' -> '{normalize_prediction(tc)}'")
