from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES_2 = ["normal", "abnormal"]
CLASS_NAMES_3 = ["normal", "HEW", "LEL"]

INPUT_VIEW_MAP = {
    "rgb_view1": ["rgb_view1"],
    "rgb_dual": ["rgb_view1", "rgb_view2"],
    "ir_only": ["ir"],
    "triple_view": ["rgb_view1", "rgb_view2", "ir"],
}


def available_input_modes() -> List[str]:
    return sorted(INPUT_VIEW_MAP.keys())


def resolve_view_keys(input_mode: str) -> List[str]:
    if input_mode not in INPUT_VIEW_MAP:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    return INPUT_VIEW_MAP[input_mode]


def load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class QwenVisFusionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        processor: Any,
        task_mode: str = "3class",
        input_mode: str = "rgb_dual",
        positive_seed: int = 42,
        random_positive: bool = True,
    ) -> None:
        super().__init__()
        self.records = load_manifest(manifest_path)
        self.processor = processor
        self.task_mode = task_mode
        self.input_mode = input_mode
        self.view_keys = resolve_view_keys(input_mode)
        self.random_positive = random_positive
        self._rng = random.Random(positive_seed)
        self.class_names = CLASS_NAMES_2 if task_mode == "2class" else CLASS_NAMES_3
        self.label_to_id = {label: idx for idx, label in enumerate(self.class_names)}
        self.same_class_other_condition_pool = self._build_positive_pool()

    def _target_label(self, record: Dict[str, Any]) -> str:
        return record["label_2c"] if self.task_mode == "2class" else record["label_3c"]

    def _build_positive_pool(self) -> Dict[int, List[int]]:
        pool: Dict[int, List[int]] = {}
        for idx, record in enumerate(self.records):
            label = self._target_label(record)
            condition_uid = record.get("condition_uid", "")
            candidates = []
            for other_idx, other_record in enumerate(self.records):
                if idx == other_idx:
                    continue
                if self._target_label(other_record) != label:
                    continue
                if other_record.get("condition_uid", "") == condition_uid:
                    continue
                candidates.append(other_idx)
            pool[idx] = candidates
        return pool

    def _select_positive_index(self, idx: int) -> Optional[int]:
        pool = self.same_class_other_condition_pool.get(idx, [])
        if not pool:
            return None
        if self.random_positive:
            return self._rng.choice(pool)
        return pool[0]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        label_name = self._target_label(record)
        images = [
            safe_load_image(record["after"][view_key])
            for view_key in self.view_keys
        ]
        processed = self.processor.image_processor.preprocess(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        image_grid_thw = processed["image_grid_thw"]

        return {
            "sample_id": record["id"],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "label_id": self.label_to_id[label_name],
            "label_name": label_name,
            "label_2c": record["label_2c"],
            "label_3c": record["label_3c"],
            "condition_uid": record.get("condition_uid", ""),
            "role": record.get("role", ""),
            "positive_index": self._select_positive_index(idx),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
        "sample_image_counts": [int(item["image_grid_thw"].shape[0]) for item in batch],
        "label_ids": torch.tensor([item["label_id"] for item in batch], dtype=torch.long),
        "label_names": [item["label_name"] for item in batch],
        "labels_2c": [item["label_2c"] for item in batch],
        "labels_3c": [item["label_3c"] for item in batch],
        "condition_uids": [item["condition_uid"] for item in batch],
        "roles": [item["role"] for item in batch],
        "positive_indices": [item.get("positive_index") for item in batch],
    }

