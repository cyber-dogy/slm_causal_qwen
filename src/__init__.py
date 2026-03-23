"""Convenience exports for the current Qwen-SLM training/evaluation package."""

__version__ = "1.1.0"
__author__ = "Qwen2-VL SLM Team"

from .dataset_qwen2vl_slm import (
    SLMDataset,
    SLMDatasetForInference,
    CLASS_NAMES_3,
    CLASS_NAMES_2,
    CLASS_NAME_TO_ID_3,
    CLASS_NAME_TO_ID_2,
    ID_TO_CLASS_NAME_3,
    ID_TO_CLASS_NAME_2,
    DEFAULT_PROMPT_3CLASS_LETTERS,
    choose_prompt,
    build_target_text,
)
from .utils_metrics import (
    compute_classification_metrics,
    normalize_prediction,
    compute_transfer_gap,
    save_metrics,
    save_predictions,
    print_metrics_table,
)

# Backward-compatible aliases for older imports.
CLASS_NAMES = CLASS_NAMES_3
CLASS_NAME_TO_ID = CLASS_NAME_TO_ID_3
ID_TO_CLASS_NAME = ID_TO_CLASS_NAME_3
DEFAULT_PROMPT = DEFAULT_PROMPT_3CLASS_LETTERS

__all__ = [
    "SLMDataset",
    "SLMDatasetForInference",
    "CLASS_NAMES",
    "CLASS_NAMES_3",
    "CLASS_NAMES_2",
    "CLASS_NAME_TO_ID",
    "CLASS_NAME_TO_ID_3",
    "CLASS_NAME_TO_ID_2",
    "ID_TO_CLASS_NAME",
    "ID_TO_CLASS_NAME_3",
    "ID_TO_CLASS_NAME_2",
    "DEFAULT_PROMPT",
    "choose_prompt",
    "build_target_text",
    "compute_classification_metrics",
    "normalize_prediction",
    "compute_transfer_gap",
    "save_metrics",
    "save_predictions",
    "print_metrics_table",
]
