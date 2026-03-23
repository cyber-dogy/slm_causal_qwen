from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES_2 = ["normal", "abnormal"]
CLASS_NAMES_3 = ["normal", "HEW", "LEL"]
CLASS_TO_ID_2 = {name: idx for idx, name in enumerate(CLASS_NAMES_2)}
CLASS_TO_ID_3 = {name: idx for idx, name in enumerate(CLASS_NAMES_3)}

INPUT_MODE_TO_VIEW_KEYS: Dict[str, List[str]] = {
    "triple_view": ["rgb_view1", "rgb_view2", "ir"],
    "rgb_dual": ["rgb_view1", "rgb_view2"],
    "rgb_view1": ["rgb_view1"],
    "rgb_view2": ["rgb_view2"],
    "ir_only": ["ir"],
}


def available_input_modes() -> List[str]:
    return sorted(INPUT_MODE_TO_VIEW_KEYS.keys())


def resolve_view_keys(input_mode: str) -> List[str]:
    if input_mode not in INPUT_MODE_TO_VIEW_KEYS:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    return list(INPUT_MODE_TO_VIEW_KEYS[input_mode])


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    ops: List[Any] = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(ops)


class ManifestMultimodalDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        target_mode: str = "3class",
        image_size: int = 224,
        train: bool = False,
        input_mode: str = "triple_view",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.target_mode = target_mode
        self.input_mode = input_mode
        self.view_keys = resolve_view_keys(input_mode)
        self.transform = build_transforms(image_size=image_size, train=train)
        self.rows = [json.loads(line) for line in self.manifest_path.open("r", encoding="utf-8") if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        images = torch.stack(
            [self._load_image(row["after"][view_key]) for view_key in self.view_keys],
            dim=0,
        )

        label_name = row["label_2c"] if self.target_mode == "2class" else row["label_3c"]
        label_id = CLASS_TO_ID_2[label_name] if self.target_mode == "2class" else CLASS_TO_ID_3[label_name]

        return {
            "id": row["id"],
            "images": images,
            "view_keys": list(self.view_keys),
            "input_mode": self.input_mode,
            "label_name": label_name,
            "label_id": label_id,
            "label_2c": row["label_2c"],
            "label_3c": row["label_3c"],
            "condition_uid": row["condition_uid"],
            "condition_id": row["condition_id"],
            "role": row.get("role", ""),
            "metadata": row.get("metadata", {}),
        }
