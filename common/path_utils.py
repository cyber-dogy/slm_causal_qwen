from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def expand_env_placeholders(value: str) -> str:
    expanded = value.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
    expanded = os.path.expandvars(expanded)
    expanded = os.path.expanduser(expanded)
    unresolved = re.findall(r"\$\{[^}]+\}", expanded)
    if unresolved:
        missing = ", ".join(sorted(set(unresolved)))
        raise ValueError(f"Unresolved environment placeholders: {missing}")
    return expanded


def expand_mapping(payload: Any) -> Any:
    if isinstance(payload, str):
        return expand_env_placeholders(payload)
    if isinstance(payload, list):
        return [expand_mapping(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(expand_mapping(item) for item in payload)
    if isinstance(payload, Mapping):
        return {key: expand_mapping(value) for key, value in payload.items()}
    return payload


def load_json_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return expand_mapping(payload)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return expand_mapping(payload)


def resolve_path(path_like: str | Path) -> Path:
    return Path(expand_env_placeholders(str(path_like))).resolve()
