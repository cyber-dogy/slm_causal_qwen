from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.path_utils import load_yaml_config


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/demo.local.yaml"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "configs/demo.example.yaml"


def _merge(defaults: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_demo_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    candidate = Path(
        config_path
        or os.environ.get("SLM_CAUSAL_QWEN_DEMO_CONFIG", str(DEFAULT_CONFIG_PATH))
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"Demo config not found: {candidate}. Copy {EXAMPLE_CONFIG_PATH} to "
            f"{DEFAULT_CONFIG_PATH} and fill in your local paths."
        )

    payload = load_yaml_config(candidate)
    defaults: Dict[str, Any] = {
        "specialized": {
            "fold": "best",
            "device": "auto",
            "question_hint": "请判断该样本属于 normal、HEW 还是 LEL，并说明是否异常。",
        },
        "general": {
            "device": "auto",
            "model_dtype": "bfloat16",
            "max_new_tokens": 256,
            "do_sample": False,
            "system_prompt": "你是一个工业视觉问答助手，请基于输入图像给出谨慎、专业、简洁的分析。",
        },
        "ui": {
            "host": "0.0.0.0",
            "port": 7861,
            "share": False,
            "title": "SLM Defect VQA Comparator",
            "example_presets": [],
        },
    }
    return _merge(defaults, payload)
