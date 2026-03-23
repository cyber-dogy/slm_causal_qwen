# -*- coding: utf-8 -*-
"""
Qwen2-VL SLM 缺陷检测 - 双用途数据集模块

功能:
1. 作为模块被 import: 提供 SLMDataset, SLMDatasetForInference, get_data_collator
2. 作为脚本直接运行: 从 metadata 构建 manifest 并进行 train/val/test 划分

用法:
    # 模式A: 作为模块使用
    from dataset_qwen2vl_slm import SLMDataset, get_data_collator
    
    # 模式B: 作为脚本运行构建 manifest
    python src/dataset_qwen2vl_slm.py --build \
        --data1_dir "/home/gjw/code/SLM_data/Causal_Image_Data" \
        --data2_dir "/home/gjw/code/SLM_data/Causal_Image_Data2" \
        --output_root "/home/gjw/code/SLM_data/processed_qwen" \
        --val_ratio 0.25 \
        --seed 42
"""

import copy
import json
import logging
import os
import random
import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageChops
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义
# =============================================================================

CLASS_NAMES_3 = ["normal", "HEW", "LEL"]
CLASS_NAME_TO_ID_3 = {"normal": 0, "HEW": 1, "LEL": 2}
ID_TO_CLASS_NAME_3 = {0: "normal", 1: "HEW", 2: "LEL"}

CLASS_NAMES_2 = ["normal", "abnormal"]
CLASS_NAME_TO_ID_2 = {"normal": 0, "abnormal": 1}
ID_TO_CLASS_NAME_2 = {0: "normal", 1: "abnormal"}

# 原始标签到三分类映射
RAW_TO_3CLASS = {
    "normal": "normal",
    "HEW": "HEW",
    "HEV": "HEW",
    "LEL": "LEL",
}

# token 风格映射
CLASS_TOKEN_MAP_3 = {
    "letters": {"normal": "A", "HEW": "B", "LEL": "C"},
    "words": {"normal": "normal", "HEW": "HEW", "LEL": "LEL"},
}
CLASS_TOKEN_MAP_2 = {
    "letters": {"normal": "N", "abnormal": "Y"},
    "words": {"normal": "normal", "abnormal": "abnormal"},
}

DEFAULT_PROMPT_3CLASS_LETTERS = (
    "Given these three monitoring images from an SLM process "
    "(RGB view1 after, RGB view2 after, IR after), classify the defect state. "
    "Output only one token: A for normal, B for HEW, C for LEL."
)

DEFAULT_PROMPT_3CLASS_WORDS = (
    "Given these three monitoring images from an SLM process "
    "(RGB view1 after, RGB view2 after, IR after), classify the defect state into one of: "
    "normal, HEW, LEL. Output only the class name."
)

DEFAULT_PROMPT_2CLASS_LETTERS = (
    "Given these three monitoring images from an SLM process "
    "(RGB view1 after, RGB view2 after, IR after), classify the sample as normal or abnormal. "
    "Output only one token: N for normal, Y for abnormal."
)

DEFAULT_PROMPT_2CLASS_WORDS = (
    "Given these three monitoring images from an SLM process "
    "(RGB view1 after, RGB view2 after, IR after), classify the sample as normal or abnormal. "
    "Output only one word: normal or abnormal."
)

DEFAULT_PROMPT_JOINT_LETTERS = (
    "Given these three monitoring images from an SLM process "
    "(RGB view1 after, RGB view2 after, IR after), output exactly two tokens separated by a single space. "
    "First token is the binary class: N for normal, Y for abnormal. "
    "Second token is the three-class label: A for normal, B for HEW, C for LEL."
)

DEFAULT_PROMPT_JOINT_WORDS = (
    "Given these three monitoring images from an SLM process "
    "(RGB view1 after, RGB view2 after, IR after), output exactly two labels separated by a single space. "
    "First label is binary: normal or abnormal. "
    "Second label is three-class: normal, HEW, or LEL."
)


# =============================================================================
# 基础工具函数
# =============================================================================

def load_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """加载 JSONL 或 JSON manifest。"""
    manifest_path = str(manifest_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    samples: List[Dict[str, Any]] = []
    suffix = Path(manifest_path).suffix.lower()

    if suffix == ".json":
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        else:
            raise ValueError(f"不支持的 JSON manifest 结构: {manifest_path}")
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"解析 {manifest_path} 第 {line_num} 行失败: {e}")

    logger.info("从 %s 加载了 %d 个样本", manifest_path, len(samples))
    return samples


def safe_load_image(
    image_path: Union[str, Path],
    max_size: Tuple[int, int] = (448, 448),
    strict: bool = True,
) -> Image.Image:
    """从路径加载图像。strict=True 时，失败会抛异常。"""
    image_path = str(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像未找到: {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        if strict:
            raise RuntimeError(f"加载图像失败: {image_path} | {e}") from e
        logger.error("加载图像失败: %s | %s", image_path, e)
        return Image.new("RGB", (224, 224), color="gray")

    if img.width > max_size[0] or img.height > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    return img


def build_absdiff_image(
    before_img: Optional[Image.Image],
    after_img: Optional[Image.Image],
) -> Optional[Image.Image]:
    """计算绝对差分图，仅作为训练辅助视图。"""
    if before_img is None or after_img is None:
        return None

    if before_img.size != after_img.size:
        before_img = before_img.resize(after_img.size, Image.Resampling.BILINEAR)

    diff = ImageChops.difference(after_img, before_img)
    return diff


def ensure_list_of_three(x: Any) -> List[str]:
    """将输入规范成长度为 3 的列表。"""
    if isinstance(x, list) and len(x) == 3:
        return [str(v) for v in x]
    raise ValueError(f"期望长度为 3 的列表，实际得到: {type(x)} | {x}")


def extract_after_paths(record: Dict[str, Any]) -> List[str]:
    """兼容旧/新 manifest 提取 after 三图。顺序: rgb_view1, rgb_view2, ir"""
    if "images_after" in record:
        return ensure_list_of_three(record["images_after"])

    if "after" in record and isinstance(record["after"], dict):
        return [
            str(record["after"]["rgb_view1"]),
            str(record["after"]["rgb_view2"]),
            str(record["after"]["ir"]),
        ]

    # 兼容扁平字段
    keys = ["rgb_view1_after_abs", "rgb_view2_after_abs", "ir_view_after_abs"]
    if all(k in record for k in keys):
        return [str(record[k]) for k in keys]

    raise ValueError(f"样本 {record.get('id', 'unknown')} 缺少 after 图像字段")


def extract_before_paths(record: Dict[str, Any]) -> List[str]:
    """兼容旧/新 manifest 提取 before 三图。若缺失则返回空列表。"""
    if "images_before" in record:
        paths = record["images_before"]
        if isinstance(paths, list):
            return [str(v) for v in paths]

    if "before" in record and isinstance(record["before"], dict):
        return [
            str(record["before"].get("rgb_view1", "")),
            str(record["before"].get("rgb_view2", "")),
            str(record["before"].get("ir", "")),
        ]

    keys = ["rgb_view1_before_abs", "rgb_view2_before_abs", "ir_view_before_abs"]
    if any(k in record for k in keys):
        return [str(record.get(k, "")) for k in keys]

    return []


def normalize_raw_label_to_3class(raw_label: str) -> str:
    raw_label = str(raw_label).strip()
    if raw_label not in RAW_TO_3CLASS:
        raise ValueError(f"无法映射到三分类的原始标签: {raw_label}")
    return RAW_TO_3CLASS[raw_label]


def derive_2class_from_3class(label_3c: str) -> str:
    return "normal" if label_3c == "normal" else "abnormal"


def choose_prompt(target_mode: str, token_style: str) -> str:
    if target_mode == "3class":
        return (
            DEFAULT_PROMPT_3CLASS_LETTERS
            if token_style == "letters"
            else DEFAULT_PROMPT_3CLASS_WORDS
        )
    if target_mode == "2class":
        return (
            DEFAULT_PROMPT_2CLASS_LETTERS
            if token_style == "letters"
            else DEFAULT_PROMPT_2CLASS_WORDS
        )
    if target_mode == "joint":
        return (
            DEFAULT_PROMPT_JOINT_LETTERS
            if token_style == "letters"
            else DEFAULT_PROMPT_JOINT_WORDS
        )
    raise ValueError(f"不支持的 target_mode: {target_mode}")


def build_target_text(
    label_3c: str,
    label_2c: str,
    target_mode: str = "3class",
    token_style: str = "letters",
) -> str:
    if target_mode == "3class":
        return CLASS_TOKEN_MAP_3[token_style][label_3c]
    if target_mode == "2class":
        return CLASS_TOKEN_MAP_2[token_style][label_2c]
    if target_mode == "joint":
        return (
            f"{CLASS_TOKEN_MAP_2[token_style][label_2c]} "
            f"{CLASS_TOKEN_MAP_3[token_style][label_3c]}"
        )
    raise ValueError(f"不支持的 target_mode: {target_mode}")


def validate_sample_record(
    record: Dict[str, Any],
    require_images: bool = True,
    allow_missing_before: bool = True,
) -> Tuple[bool, str]:
    """验证样本结构。"""
    if "id" not in record:
        return False, "缺少 'id'"

    try:
        after_paths = extract_after_paths(record)
    except Exception as e:
        return False, f"after 图像解析失败: {e}"

    if len(after_paths) != 3:
        return False, f"after 图像数不为 3: {after_paths}"

    if require_images:
        for p in after_paths:
            if not os.path.exists(p):
                return False, f"after 图像不存在: {p}"

    before_paths = extract_before_paths(record)
    if before_paths:
        if len(before_paths) != 3:
            return False, f"before 图像数不为 3: {before_paths}"
        if require_images:
            for p in before_paths:
                if p and (not os.path.exists(p)):
                    if allow_missing_before:
                        logger.warning("before 图像不存在但允许缺失: %s", p)
                    else:
                        return False, f"before 图像不存在: {p}"

    raw_label = (
        record.get("label_raw")
        or record.get("defect_label_type")
        or record.get("label_3c")
        or record.get("label")
    )
    if raw_label is None:
        return False, "缺少标签字段"

    try:
        _ = normalize_raw_label_to_3class(str(raw_label))
    except Exception as e:
        return False, str(e)

    return True, ""


# =============================================================================
# 对话构建
# =============================================================================

def build_conversation(
    images_after: List[str],
    prompt: str,
    target_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """仅用 after 三图构建 Qwen2-VL 对话。before 不进入主对话输入。"""
    user_content: List[Dict[str, Any]] = []

    for img_path in images_after:
        user_content.append({"type": "image", "image": str(img_path)})

    user_content.append({"type": "text", "text": prompt})

    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_content}]

    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )

    return messages


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class SLMSample:
    id: str
    split: str
    images_after: List[str]
    images_before: List[str]

    label_raw: str
    label_3c: str
    label_3c_id: int
    label_2c: str
    label_2c_id: int

    prompt: str
    metadata: Dict[str, Any]

    dataset_domain: str = ""
    shape_domain: str = ""
    condition_id: str = ""
    condition_uid: str = ""
    layer_id: int = -1
    role: str = ""
    defect_label_type_raw: str = ""

    target_text_3c: str = ""
    target_text_2c: str = ""
    target_text_joint: str = ""

    def __post_init__(self):
        self.dataset_domain = str(
            self.metadata.get("dataset_domain", self.dataset_domain or "")
        )
        self.shape_domain = str(self.metadata.get("shape_domain", self.shape_domain or ""))
        self.condition_id = str(self.metadata.get("condition_id", self.condition_id or ""))
        self.condition_uid = str(
            self.metadata.get(
                "condition_uid",
                self.condition_uid or (
                    f"{self.dataset_domain}__{self.condition_id}"
                    if self.dataset_domain and self.condition_id else self.condition_id
                ),
            )
        )
        self.layer_id = int(self.metadata.get("layer_id", self.layer_id if self.layer_id != -1 else -1))
        self.role = str(self.metadata.get("role", self.role or ""))
        self.defect_label_type_raw = str(
            self.metadata.get("defect_label_type_raw", self.defect_label_type_raw or self.label_raw)
        )

        if not self.target_text_3c:
            self.target_text_3c = build_target_text(self.label_3c, self.label_2c, "3class", "letters")
        if not self.target_text_2c:
            self.target_text_2c = build_target_text(self.label_3c, self.label_2c, "2class", "letters")
        if not self.target_text_joint:
            self.target_text_joint = build_target_text(self.label_3c, self.label_2c, "joint", "letters")


# =============================================================================
# 主数据集
# =============================================================================

class SLMDataset(Dataset):
    """
    兼容 naive SFT / causal training / inference 的统一数据集。

    关键参数
    --------
    task_mode: "naive_sft", "causal_train", "inference"
    target_mode: "3class", "2class", "joint"
    token_style: "letters", "words"
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        processor: Optional[Any],
        prompt: Optional[str] = None,
        max_length: int = 2048,
        split: Optional[str] = None,
        require_images: bool = True,
        skip_invalid: bool = True,
        allow_missing_before: bool = True,
        task_mode: str = "naive_sft",
        target_mode: str = "3class",
        token_style: str = "letters",
        include_before_images: bool = False,
        include_change_images: bool = False,
        return_raw_images: bool = False,
        strict_image_loading: bool = True,
        max_image_size: Tuple[int, int] = (448, 448),
        neighbor_window: int = 1,
        random_positive: bool = True,
        positive_seed: int = 42,
    ):
        self.manifest_path = str(manifest_path)
        self.processor = processor
        self.max_length = max_length
        self.require_images = require_images
        self.skip_invalid = skip_invalid
        self.allow_missing_before = allow_missing_before
        self.task_mode = task_mode
        self.target_mode = target_mode
        self.token_style = token_style
        self.include_before_images = include_before_images or (task_mode == "causal_train")
        self.include_change_images = include_change_images or (task_mode == "causal_train")
        self.return_raw_images = return_raw_images
        self.strict_image_loading = strict_image_loading
        self.max_image_size = max_image_size
        self.neighbor_window = neighbor_window
        self.random_positive = random_positive
        self._rng = random.Random(positive_seed)

        if prompt is None:
            self.prompt = choose_prompt(target_mode=target_mode, token_style=token_style)
        else:
            self.prompt = prompt

        raw_records = load_manifest(self.manifest_path)
        self.samples: List[SLMSample] = []
        invalid_count = 0

        for raw in raw_records:
            sample_split = split or raw.get("split", "unknown")
            if split is not None and sample_split != split:
                continue

            is_valid, err = validate_sample_record(
                raw,
                require_images=require_images,
                allow_missing_before=allow_missing_before,
            )
            if not is_valid:
                invalid_count += 1
                msg = f"跳过无效样本 {raw.get('id', 'unknown')}: {err}"
                if skip_invalid:
                    logger.warning(msg)
                    continue
                raise ValueError(msg)

            try:
                sample = self._parse_record_to_sample(raw, sample_split)
                self.samples.append(sample)
            except Exception as e:
                invalid_count += 1
                msg = f"样本解析失败 {raw.get('id', 'unknown')}: {e}"
                if skip_invalid:
                    logger.warning(msg)
                    continue
                raise

        logger.info(
            "数据集加载完成: %d 个有效样本，跳过 %d 个无效样本",
            len(self.samples), invalid_count
        )

        self._build_group_indices()
        self._log_distribution()

    def _parse_record_to_sample(self, raw: Dict[str, Any], sample_split: str) -> SLMSample:
        images_after = extract_after_paths(raw)
        images_before = extract_before_paths(raw)

        metadata = dict(raw.get("metadata", {}))
        for key in [
            "dataset_domain", "shape_domain", "condition_id", "condition_uid",
            "layer_id", "role", "power_w", "speed_mms", "spacing_mm",
            "scan_strategy", "energy_density", "batch_id"
        ]:
            if key in raw and key not in metadata:
                metadata[key] = raw[key]

        label_raw = (
            raw.get("label_raw")
            or raw.get("defect_label_type")
            or raw.get("label_3c")
            or raw.get("label")
        )
        label_raw = str(label_raw)

        if raw.get("label_3c") in CLASS_NAME_TO_ID_3:
            label_3c = str(raw["label_3c"])
        else:
            label_3c = normalize_raw_label_to_3class(label_raw)

        if raw.get("label_2c") in CLASS_NAME_TO_ID_2:
            label_2c = str(raw["label_2c"])
        else:
            label_2c = derive_2class_from_3class(label_3c)

        label_3c_id = CLASS_NAME_TO_ID_3[label_3c]
        label_2c_id = CLASS_NAME_TO_ID_2[label_2c]

        prompt = raw.get("prompt", self.prompt)

        return SLMSample(
            id=str(raw["id"]),
            split=str(sample_split),
            images_after=images_after,
            images_before=images_before,
            label_raw=label_raw,
            label_3c=label_3c,
            label_3c_id=label_3c_id,
            label_2c=label_2c,
            label_2c_id=label_2c_id,
            prompt=str(prompt),
            metadata=metadata,
            target_text_3c=build_target_text(label_3c, label_2c, "3class", self.token_style),
            target_text_2c=build_target_text(label_3c, label_2c, "2class", self.token_style),
            target_text_joint=build_target_text(label_3c, label_2c, "joint", self.token_style),
        )

    def _build_group_indices(self) -> None:
        self.indices_by_label_3c: Dict[str, List[int]] = {k: [] for k in CLASS_NAMES_3}
        self.indices_by_label_2c: Dict[str, List[int]] = {k: [] for k in CLASS_NAMES_2}
        self.indices_by_condition: Dict[str, List[int]] = {}

        for idx, s in enumerate(self.samples):
            self.indices_by_label_3c[s.label_3c].append(idx)
            self.indices_by_label_2c[s.label_2c].append(idx)
            self.indices_by_condition.setdefault(s.condition_uid, []).append(idx)

        self.same_class_other_condition_pool: Dict[int, List[int]] = {}
        for idx, s in enumerate(self.samples):
            pool = []
            for j in self.indices_by_label_3c[s.label_3c]:
                if j == idx:
                    continue
                if self.samples[j].condition_uid != s.condition_uid:
                    pool.append(j)
            self.same_class_other_condition_pool[idx] = pool

        self.same_condition_neighbor_pool: Dict[int, List[int]] = {}
        for cond, ids in self.indices_by_condition.items():
            ids_sorted = sorted(ids, key=lambda k: self.samples[k].layer_id)
            for pos, idx in enumerate(ids_sorted):
                left = max(0, pos - self.neighbor_window)
                right = min(len(ids_sorted), pos + self.neighbor_window + 1)
                pool = [ids_sorted[p] for p in range(left, right) if ids_sorted[p] != idx]
                self.same_condition_neighbor_pool[idx] = pool

    def _log_distribution(self) -> None:
        if len(self.samples) == 0:
            logger.warning("数据集为空")
            return

        logger.info("三分类分布:")
        for name in CLASS_NAMES_3:
            c = len(self.indices_by_label_3c.get(name, []))
            logger.info("  %s: %d (%.1f%%)", name, c, 100.0 * c / len(self.samples))

        logger.info("二分类分布:")
        for name in CLASS_NAMES_2:
            c = len(self.indices_by_label_2c.get(name, []))
            logger.info("  %s: %d (%.1f%%)", name, c, 100.0 * c / len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def _select_positive_index(self, idx: int) -> Optional[int]:
        pool = self.same_class_other_condition_pool.get(idx, [])
        if not pool:
            return None
        if self.random_positive:
            return self._rng.choice(pool)
        return pool[0]

    def _active_target_text(self, sample: SLMSample) -> str:
        if self.target_mode == "3class":
            return sample.target_text_3c
        if self.target_mode == "2class":
            return sample.target_text_2c
        if self.target_mode == "joint":
            return sample.target_text_joint
        raise ValueError(f"不支持的 target_mode: {self.target_mode}")

    def _active_label_name(self, sample: SLMSample) -> str:
        if self.target_mode == "3class":
            return sample.label_3c
        if self.target_mode == "2class":
            return sample.label_2c
        if self.target_mode == "joint":
            return sample.target_text_joint
        raise ValueError(f"不支持的 target_mode: {self.target_mode}")

    def _active_label_id(self, sample: SLMSample) -> int:
        if self.target_mode == "3class":
            return sample.label_3c_id
        if self.target_mode == "2class":
            return sample.label_2c_id
        if self.target_mode == "joint":
            return sample.label_3c_id
        raise ValueError(f"不支持的 target_mode: {self.target_mode}")

    def _encode_with_processor(
        self,
        after_images_pil: List[Image.Image],
        prompt: str,
        target_text: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        """用 processor 构造 multimodal 输入。labels 只监督 assistant 回复。"""
        if self.processor is None:
            raise ValueError("processor 为 None，无法进行 tokenization/encoding")

        full_messages = build_conversation(
            images_after=["__DUMMY__"] * 3,
            prompt=prompt,
            target_text=target_text,
        )

        prefix_messages = build_conversation(
            images_after=["__DUMMY__"] * 3,
            prompt=prompt,
            target_text=None,
        )

        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_text = self.processor.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_inputs = self.processor(
            text=[full_text],
            images=after_images_pil,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        prefix_inputs = self.processor(
            text=[prefix_text],
            images=after_images_pil,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        encoded: Dict[str, torch.Tensor] = {}
        for key, value in full_inputs.items():
            if isinstance(value, torch.Tensor):
                encoded[key] = value.squeeze(0)
            else:
                encoded[key] = value

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prefix_len = int(prefix_inputs["input_ids"].shape[-1])

        labels = input_ids.clone()
        prefix_len = min(prefix_len, labels.shape[0])
        labels[:prefix_len] = -100

        if torch.all(labels == -100) and labels.numel() > 0:
            labels[-1] = input_ids[-1]

        encoded["labels"] = labels
        encoded["attention_mask"] = attention_mask

        return encoded

    def _load_after_images(self, sample: SLMSample) -> List[Image.Image]:
        return [
            safe_load_image(p, max_size=self.max_image_size, strict=self.strict_image_loading)
            for p in sample.images_after
        ]

    def _load_before_images(self, sample: SLMSample) -> List[Optional[Image.Image]]:
        images: List[Optional[Image.Image]] = []
        if not sample.images_before:
            return [None, None, None]

        for p in sample.images_before:
            if (not p) or (not os.path.exists(p)):
                if self.allow_missing_before:
                    images.append(None)
                else:
                    raise FileNotFoundError(f"before 图像不存在: {p}")
            else:
                images.append(
                    safe_load_image(
                        p,
                        max_size=self.max_image_size,
                        strict=self.strict_image_loading,
                    )
                )

        while len(images) < 3:
            images.append(None)

        return images[:3]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        after_images_pil = self._load_after_images(sample)

        before_images_pil: List[Optional[Image.Image]] = [None, None, None]
        change_images_pil: List[Optional[Image.Image]] = [None, None, None]

        if self.include_before_images or self.include_change_images or self.task_mode == "causal_train":
            before_images_pil = self._load_before_images(sample)

        if self.include_change_images or self.task_mode == "causal_train":
            change_images_pil = [
                build_absdiff_image(b, a)
                for b, a in zip(before_images_pil, after_images_pil)
            ]

        item: Dict[str, Any] = {
            "sample_id": sample.id,
            "split": sample.split,
            "label": self._active_label_name(sample),
            "label_id": self._active_label_id(sample),
            "label_3c": sample.label_3c,
            "label_3c_id": sample.label_3c_id,
            "label_2c": sample.label_2c,
            "label_2c_id": sample.label_2c_id,
            "target_text": self._active_target_text(sample),
            "metadata": sample.metadata,
            "dataset_domain": sample.dataset_domain,
            "shape_domain": sample.shape_domain,
            "condition_id": sample.condition_id,
            "condition_uid": sample.condition_uid,
            "layer_id": sample.layer_id,
            "role": sample.role,
            "images_after": sample.images_after,
            "images_before": sample.images_before,
            "positive_pool_indices": self.same_class_other_condition_pool.get(idx, []),
            "neighbor_pool_indices": self.same_condition_neighbor_pool.get(idx, []),
        }

        if self.processor is None:
            item["after_images_pil"] = after_images_pil if self.return_raw_images else None
            item["before_images_pil"] = before_images_pil if self.return_raw_images else None
            item["change_images_pil"] = change_images_pil if self.return_raw_images else None
            return item

        encoded = self._encode_with_processor(
            after_images_pil=after_images_pil,
            prompt=sample.prompt,
            target_text=self._active_target_text(sample) if self.task_mode != "inference" else None,
        )
        item.update(encoded)

        if self.task_mode == "causal_train":
            pos_idx = self._select_positive_index(idx)
            item["positive_index"] = pos_idx
            item["positive_sample_id"] = self.samples[pos_idx].id if pos_idx is not None else None

        if self.include_before_images or self.task_mode == "causal_train":
            item["before_images_pil"] = before_images_pil if self.return_raw_images else None

        if self.include_change_images or self.task_mode == "causal_train":
            item["change_images_pil"] = change_images_pil if self.return_raw_images else None

        item["has_before"] = any(p for p in sample.images_before)
        item["has_change"] = any(img is not None for img in change_images_pil)

        return item

    def get_subset(self, condition: Callable[[SLMSample], bool]) -> "SLMDataset":
        new_dataset = copy.copy(self)
        new_dataset.samples = [s for s in self.samples if condition(s)]
        new_dataset._build_group_indices()
        new_dataset._log_distribution()
        return new_dataset

    def get_by_dataset_domain(self, domain: str) -> "SLMDataset":
        return self.get_subset(lambda s: s.dataset_domain == domain)

    def get_by_role(self, role_prefix: str) -> "SLMDataset":
        return self.get_subset(lambda s: s.role.startswith(role_prefix))

    def get_by_label_3c(self, label_3c: str) -> "SLMDataset":
        return self.get_subset(lambda s: s.label_3c == label_3c)

    def get_by_label_2c(self, label_2c: str) -> "SLMDataset":
        return self.get_subset(lambda s: s.label_2c == label_2c)

    def get_by_condition_uid(self, condition_uid: str) -> "SLMDataset":
        return self.get_subset(lambda s: s.condition_uid == condition_uid)


# =============================================================================
# 推理专用数据集
# =============================================================================

class SLMDatasetForInference(SLMDataset):
    """推理版数据集：不构造带标签的 assistant 回复。"""
    def __init__(
        self,
        manifest_path: Union[str, Path],
        processor: Optional[Any],
        prompt: Optional[str] = None,
        max_length: int = 2048,
        split: Optional[str] = None,
        require_images: bool = True,
        skip_invalid: bool = True,
        allow_missing_before: bool = True,
        target_mode: str = "3class",
        token_style: str = "letters",
        include_before_images: bool = False,
        include_change_images: bool = False,
        return_raw_images: bool = False,
        strict_image_loading: bool = True,
        max_image_size: Tuple[int, int] = (448, 448),
    ):
        super().__init__(
            manifest_path=manifest_path,
            processor=processor,
            prompt=prompt,
            max_length=max_length,
            split=split,
            require_images=require_images,
            skip_invalid=skip_invalid,
            allow_missing_before=allow_missing_before,
            task_mode="inference",
            target_mode=target_mode,
            token_style=token_style,
            include_before_images=include_before_images,
            include_change_images=include_change_images,
            return_raw_images=return_raw_images,
            strict_image_loading=strict_image_loading,
            max_image_size=max_image_size,
        )


# =============================================================================
# Data collator
# =============================================================================

def get_data_collator(processor: Any, padding: bool = True):
    """适配 Qwen2-VL 的 collate_fn。"""
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    def _pad_sequence(
        seqs: List[torch.Tensor],
        padding_value: int,
    ) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            seqs,
            batch_first=True,
            padding_value=padding_value,
        )

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated: Dict[str, Any] = {}

        tensor_keys = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]

        for key in tensor_keys:
            values = [item[key] for item in batch if key in item and item[key] is not None]
            if not values:
                continue

            if key == "pixel_values":
                collated[key] = torch.stack(values, dim=0)
            elif key == "image_grid_thw":
                collated[key] = torch.cat(values, dim=0)
            elif key == "attention_mask":
                if padding:
                    collated[key] = _pad_sequence(values, padding_value=0)
                else:
                    collated[key] = torch.stack(values, dim=0)
            elif key == "labels":
                if padding:
                    collated[key] = _pad_sequence(values, padding_value=-100)
                else:
                    collated[key] = torch.stack(values, dim=0)
            else:  # input_ids
                if padding:
                    collated[key] = _pad_sequence(values, padding_value=pad_token_id)
                else:
                    collated[key] = torch.stack(values, dim=0)

        # 兼容旧脚本字段
        collated["sample_ids"] = [item["sample_id"] for item in batch]
        collated["labels_str"] = [item["label"] for item in batch]
        collated["label_ids"] = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)

        # 新增显式二/三分类标签
        collated["label_ids_3c"] = torch.tensor([item["label_3c_id"] for item in batch], dtype=torch.long)
        collated["label_ids_2c"] = torch.tensor([item["label_2c_id"] for item in batch], dtype=torch.long)
        collated["labels_str_3c"] = [item["label_3c"] for item in batch]
        collated["labels_str_2c"] = [item["label_2c"] for item in batch]

        # 元数据
        collated["metadata"] = [item.get("metadata", {}) for item in batch]
        collated["condition_uids"] = [item.get("condition_uid", "") for item in batch]
        collated["dataset_domains"] = [item.get("dataset_domain", "") for item in batch]
        collated["shape_domains"] = [item.get("shape_domain", "") for item in batch]
        collated["roles"] = [item.get("role", "") for item in batch]

        # 因果训练辅助字段
        collated["images_before"] = [item.get("images_before", []) for item in batch]
        collated["positive_indices"] = [item.get("positive_index", None) for item in batch]
        collated["positive_sample_ids"] = [item.get("positive_sample_id", None) for item in batch]
        collated["positive_pool_indices"] = [item.get("positive_pool_indices", []) for item in batch]
        collated["neighbor_pool_indices"] = [item.get("neighbor_pool_indices", []) for item in batch]
        collated["has_before"] = [item.get("has_before", False) for item in batch]
        collated["has_change"] = [item.get("has_change", False) for item in batch]

        # 原始 PIL 图像
        if any("before_images_pil" in item for item in batch):
            collated["before_images_pil"] = [item.get("before_images_pil", None) for item in batch]
        if any("change_images_pil" in item for item in batch):
            collated["change_images_pil"] = [item.get("change_images_pil", None) for item in batch]

        return collated

    return collate_fn


def get_class_weights(dataset: SLMDataset, label_mode: str = "3class") -> torch.Tensor:
    """为不平衡数据集计算类别权重。"""
    if label_mode == "3class":
        counts = {name: 0 for name in CLASS_NAMES_3}
        for sample in dataset.samples:
            counts[sample.label_3c] += 1
        total = sum(counts.values())
        n_classes = len(CLASS_NAMES_3)
        weights = [
            total / (n_classes * counts[name]) if counts[name] > 0 else 1.0
            for name in CLASS_NAMES_3
        ]
        return torch.tensor(weights, dtype=torch.float32)

    if label_mode == "2class":
        counts = {name: 0 for name in CLASS_NAMES_2}
        for sample in dataset.samples:
            counts[sample.label_2c] += 1
        total = sum(counts.values())
        n_classes = len(CLASS_NAMES_2)
        weights = [
            total / (n_classes * counts[name]) if counts[name] > 0 else 1.0
            for name in CLASS_NAMES_2
        ]
        return torch.tensor(weights, dtype=torch.float32)

    raise ValueError(f"不支持的 label_mode: {label_mode}")


def create_subset_from_metadata(
    dataset: SLMDataset,
    metadata_filter: Dict[str, Any],
) -> SLMDataset:
    """按 metadata 创建子集。"""
    def condition(sample: SLMSample) -> bool:
        for key, value in metadata_filter.items():
            if sample.metadata.get(key) != value:
                return False
        return True

    return dataset.get_subset(condition)


def fold_predictions_3c_to_2c(pred_3c: Union[List[int], np.ndarray, torch.Tensor]) -> np.ndarray:
    """将三分类预测折叠为二分类: normal -> normal(0), HEW/LEL -> abnormal(1)"""
    if isinstance(pred_3c, torch.Tensor):
        pred_3c = pred_3c.detach().cpu().numpy()
    pred_3c = np.asarray(pred_3c)
    out = np.where(pred_3c == CLASS_NAME_TO_ID_3["normal"], 0, 1)
    return out.astype(np.int64)


# =============================================================================
# 模式 B: 作为脚本运行 - 从 metadata 构建 manifest
# =============================================================================

def load_metadata_from_file(metadata_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """从 JSON 或 CSV 加载 metadata。"""
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata 文件不存在: {metadata_path}")
    
    suffix = metadata_path.suffix.lower()
    
    if suffix == ".json":
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "samples" in data:
            return data["samples"]
        else:
            raise ValueError(f"不支持的 JSON 结构: {metadata_path}")
    
    elif suffix == ".csv":
        import csv
        records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值类型
                for key in ["layer_id", "batch_id", "defect_label_binary"]:
                    if key in row:
                        try:
                            row[key] = int(row[key])
                        except (ValueError, TypeError):
                            pass
                for key in ["power_w", "speed_mms", "spacing_mm", "energy_density"]:
                    if key in row:
                        try:
                            row[key] = float(row[key])
                        except (ValueError, TypeError):
                            pass
                records.append(dict(row))
        return records
    
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def resolve_image_path(rel_path: str, dataset_dir: Union[str, Path]) -> str:
    """将相对路径解析为绝对路径。"""
    dataset_dir = Path(dataset_dir)
    
    # 如果是绝对路径，直接返回
    if os.path.isabs(rel_path):
        return rel_path
    
    # 移除开头的 ./
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    
    # 组合路径
    abs_path = dataset_dir / rel_path
    return str(abs_path.resolve())


def str2bool(value: Any) -> bool:
    """可靠地解析命令行布尔值。"""
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def _detect_metadata_path(dataset_dir: Union[str, Path]) -> Path:
    dataset_dir = Path(dataset_dir)
    metadata_json = dataset_dir / "metadata.json"
    metadata_csv = dataset_dir / "metadata.csv"

    if metadata_json.exists():
        return metadata_json
    if metadata_csv.exists():
        return metadata_csv

    raise FileNotFoundError(f"在 {dataset_dir} 下未找到 metadata.json 或 metadata.csv")


def _normalize_role(record: Dict[str, Any]) -> str:
    role = record.get("role", "unknown")
    return str(role).strip() or "unknown"


def _build_image_triplet(
    record: Dict[str, Any],
    dataset_dir: Union[str, Path],
    field_prefix: str,
) -> Dict[str, str]:
    field_map = {
        "after": [
            "rgb_view1_after_path",
            "rgb_view2_after_path",
            "ir_view_after_path",
        ],
        "before": [
            "rgb_view1_before_path",
            "rgb_view2_before_path",
            "ir_view_before_path",
        ],
    }
    if field_prefix not in field_map:
        raise ValueError(f"不支持的图像前缀: {field_prefix}")

    keys = field_map[field_prefix]
    paths = {
        "rgb_view1": resolve_image_path(record.get(keys[0], ""), dataset_dir),
        "rgb_view2": resolve_image_path(record.get(keys[1], ""), dataset_dir),
        "ir": resolve_image_path(record.get(keys[2], ""), dataset_dir),
    }

    missing = [path for path in paths.values() if (not path) or (not os.path.exists(path))]
    if missing:
        raise FileNotFoundError(
            f"{field_prefix} 图像缺失: condition={record.get('condition_id')} "
            f"layer={record.get('layer_id')} | {missing}"
        )

    return paths


def _build_sample_from_record(
    record: Dict[str, Any],
    dataset_dir: Union[str, Path],
    dataset_domain: str,
) -> Dict[str, Any]:
    condition_id = str(record["condition_id"]).strip()
    layer_id = int(record["layer_id"])
    label_raw = str(record.get("defect_label_type", "normal")).strip()
    label_3c = normalize_raw_label_to_3class(label_raw)
    label_2c = derive_2class_from_3class(label_3c)
    role = _normalize_role(record)
    condition_uid = f"{dataset_domain}__{condition_id}"

    sample_id = f"{dataset_domain}__{condition_id}__L{layer_id:04d}"
    metadata = {
        "dataset_domain": dataset_domain,
        "condition_id": condition_id,
        "condition_uid": condition_uid,
        "layer_id": layer_id,
        "role": role,
        "power_w": record.get("power_w"),
        "speed_mms": record.get("speed_mms"),
        "spacing_mm": record.get("spacing_mm"),
        "scan_strategy": record.get("scan_strategy"),
        "energy_density": record.get("energy_density"),
        "batch_id": record.get("batch_id"),
        "defect_label_type_raw": label_raw,
    }

    return {
        "id": sample_id,
        "dataset_domain": dataset_domain,
        "condition_id": condition_id,
        "condition_uid": condition_uid,
        "layer_id": layer_id,
        "role": role,
        "split": "",
        "label_raw": label_raw,
        "label_3c": label_3c,
        "label_2c": label_2c,
        "defect_label_type": label_raw,
        "defect_label_binary": int(label_2c == "abnormal"),
        "metadata": metadata,
        "after": _build_image_triplet(record, dataset_dir, "after"),
        "before": _build_image_triplet(record, dataset_dir, "before"),
    }


def _load_samples_from_dataset_dir(
    dataset_dir: Union[str, Path],
    dataset_domain: str,
) -> List[Dict[str, Any]]:
    metadata_path = _detect_metadata_path(dataset_dir)
    logger.info("加载 %s metadata: %s", dataset_domain, metadata_path)
    records = load_metadata_from_file(metadata_path)
    samples = [
        _build_sample_from_record(record, dataset_dir=dataset_dir, dataset_domain=dataset_domain)
        for record in records
    ]
    logger.info("%s 加载了 %d 条记录", dataset_domain, len(samples))
    return samples


def _clone_sample_with_split(sample: Dict[str, Any], split: str) -> Dict[str, Any]:
    cloned = copy.deepcopy(sample)
    cloned["split"] = split
    return cloned


def _shuffle_in_place(samples: List[Dict[str, Any]], rng: random.Random) -> None:
    if len(samples) > 1:
        rng.shuffle(samples)


def _compute_split_counts(
    class_count: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int]:
    if class_count < 3 and (val_ratio > 0 or test_ratio > 0):
        raise ValueError(f"类别样本数过少，无法切分 train/val/test: {class_count}")

    n_val = int(round(class_count * val_ratio)) if val_ratio > 0 else 0
    n_test = int(round(class_count * test_ratio)) if test_ratio > 0 else 0

    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if test_ratio > 0 and n_test == 0:
        n_test = 1

    while n_val + n_test >= class_count:
        if n_test > n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    if n_val + n_test >= class_count:
        raise ValueError(
            f"类别样本数不足以留出训练集: class_count={class_count}, "
            f"n_val={n_val}, n_test={n_test}"
        )

    return n_val, n_test


def _split_source_samples(
    samples: List[Dict[str, Any]],
    source_val_ratio: float,
    source_test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in CLASS_NAMES_3}

    for sample in samples:
        by_label[sample["label_3c"]].append(sample)

    missing_labels = [label for label, items in by_label.items() if not items]
    if missing_labels:
        raise ValueError(f"source 数据缺少类别，无法切分: {missing_labels}")

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    for label, label_samples in by_label.items():
        label_pool = list(label_samples)
        _shuffle_in_place(label_pool, rng)

        n_val, n_test = _compute_split_counts(
            class_count=len(label_pool),
            val_ratio=source_val_ratio,
            test_ratio=source_test_ratio,
        )

        val_slice = label_pool[:n_val]
        test_slice = label_pool[n_val:n_val + n_test]
        train_slice = label_pool[n_val + n_test:]

        val_samples.extend(_clone_sample_with_split(sample, "val") for sample in val_slice)
        test_samples.extend(_clone_sample_with_split(sample, "test") for sample in test_slice)
        train_samples.extend(_clone_sample_with_split(sample, "train") for sample in train_slice)

    _shuffle_in_place(train_samples, rng)
    _shuffle_in_place(val_samples, rng)
    _shuffle_in_place(test_samples, rng)

    return train_samples, val_samples, test_samples


def _balance_train_samples_upsample_to_max(
    samples: List[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in CLASS_NAMES_3}
    for sample in samples:
        by_label[sample["label_3c"]].append(sample)

    missing_labels = [label for label, items in by_label.items() if not items]
    if missing_labels:
        raise ValueError(f"train 数据缺少类别，无法均衡: {missing_labels}")

    target_count = max(len(items) for items in by_label.values())
    balanced: List[Dict[str, Any]] = []

    for label in CLASS_NAMES_3:
        label_samples = by_label[label]
        selected = list(label_samples)
        while len(selected) < target_count:
            selected.append(rng.choice(label_samples))

        id_counts: Dict[str, int] = {}
        for base_sample in selected:
            sample_copy = copy.deepcopy(base_sample)
            original_id = sample_copy["id"]
            duplicate_index = id_counts.get(original_id, 0)

            if duplicate_index > 0:
                sample_copy["original_id"] = original_id
                sample_copy["id"] = f"{original_id}__dup{duplicate_index:02d}"
                sample_copy["metadata"] = dict(sample_copy.get("metadata", {}))
                sample_copy["metadata"]["original_id"] = original_id

            id_counts[original_id] = duplicate_index + 1
            balanced.append(sample_copy)

    _shuffle_in_place(balanced, rng)
    return balanced


def _validate_required_sample_fields(samples: List[Dict[str, Any]], split_name: str) -> None:
    required_top_level = [
        "id", "split", "dataset_domain", "condition_id", "condition_uid",
        "layer_id", "role", "label_raw", "label_3c", "label_2c",
        "after", "before", "metadata",
    ]
    required_image_keys = ["rgb_view1", "rgb_view2", "ir"]

    for sample in samples:
        for key in required_top_level:
            if key not in sample:
                raise ValueError(f"{split_name} 样本缺少字段 {key}: {sample.get('id', 'unknown')}")

        for group_name in ["after", "before"]:
            group = sample[group_name]
            if not isinstance(group, dict):
                raise ValueError(f"{split_name} 样本 {sample['id']} 的 {group_name} 不是字典")
            for image_key in required_image_keys:
                path = group.get(image_key, "")
                if not path or (not os.path.exists(path)):
                    raise FileNotFoundError(
                        f"{split_name} 样本 {sample['id']} 的 {group_name}.{image_key} 缺失: {path}"
                    )

        metadata = sample["metadata"]
        for key in ["dataset_domain", "condition_id", "condition_uid", "layer_id", "role"]:
            if key not in metadata or metadata[key] in ("", None):
                raise ValueError(f"{split_name} 样本 {sample['id']} 缺少 metadata.{key}")


def _validate_split_integrity(
    split_name: str,
    samples: List[Dict[str, Any]],
    expected_domain: Optional[str] = None,
    require_full_3class: bool = False,
) -> None:
    if not samples:
        raise ValueError(f"{split_name} 为空")

    _validate_required_sample_fields(samples, split_name)

    if expected_domain is not None:
        observed_domains = {sample["dataset_domain"] for sample in samples}
        if observed_domains != {expected_domain}:
            raise ValueError(
                f"{split_name} 域错误，期望 {expected_domain}，实际 {sorted(observed_domains)}"
            )

    if require_full_3class:
        observed_labels = {sample["label_3c"] for sample in samples}
        missing = [label for label in CLASS_NAMES_3 if label not in observed_labels]
        if missing:
            raise ValueError(f"{split_name} 缺少类别: {missing}")

    before_after_overlap = 0
    for sample in samples:
        if all(
            sample["before"][image_key] == sample["after"][image_key]
            for image_key in ["rgb_view1", "rgb_view2", "ir"]
        ):
            before_after_overlap += 1

    if before_after_overlap > 0:
        raise ValueError(
            f"{split_name} 中存在 before 与 after 路径完全相同的样本: {before_after_overlap}"
        )


def _summarize_split(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "n_samples": 0,
            "domain_distribution": {},
            "label_3c_distribution": {},
            "label_2c_distribution": {},
            "before_available_rate": 0.0,
            "before_equals_after_rate": 0.0,
            "unique_conditions": 0,
        }

    before_available = 0
    before_equals_after = 0
    for sample in samples:
        has_before = all(sample["before"].get(key, "") for key in ["rgb_view1", "rgb_view2", "ir"])
        if has_before:
            before_available += 1
        if all(
            sample["before"][image_key] == sample["after"][image_key]
            for image_key in ["rgb_view1", "rgb_view2", "ir"]
        ):
            before_equals_after += 1

    return {
        "n_samples": len(samples),
        "domain_distribution": dict(Counter(sample["dataset_domain"] for sample in samples)),
        "label_3c_distribution": dict(Counter(sample["label_3c"] for sample in samples)),
        "label_2c_distribution": dict(Counter(sample["label_2c"] for sample in samples)),
        "before_available_rate": before_available / len(samples),
        "before_equals_after_rate": before_equals_after / len(samples),
        "unique_conditions": len({sample["condition_uid"] for sample in samples}),
    }


def _build_split_summary_rows(
    split_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    image_keys = ["rgb_view1", "rgb_view2", "ir"]

    for split_name, samples in split_map.items():
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for sample in samples:
            key = (sample["dataset_domain"], sample["label_3c"])
            stats = grouped.setdefault(
                key,
                {
                    "count": 0,
                    "before_available_count": 0,
                    "before_equals_after_count": 0,
                    "label_2c": sample["label_2c"],
                },
            )
            stats["count"] += 1

            if all(sample["before"].get(image_key, "") for image_key in image_keys):
                stats["before_available_count"] += 1

            if all(
                sample["before"][image_key] == sample["after"][image_key]
                for image_key in image_keys
            ):
                stats["before_equals_after_count"] += 1

        for (dataset_domain, label_3c), stats in sorted(grouped.items()):
            count = stats["count"]
            rows.append(
                {
                    "split": split_name,
                    "dataset_domain": dataset_domain,
                    "label_3c": label_3c,
                    "label_2c": stats["label_2c"],
                    "count": count,
                    "before_available_count": stats["before_available_count"],
                    "before_available_rate": (
                        stats["before_available_count"] / count if count > 0 else 0.0
                    ),
                    "before_equals_after_count": stats["before_equals_after_count"],
                    "before_equals_after_rate": (
                        stats["before_equals_after_count"] / count if count > 0 else 0.0
                    ),
                }
            )

    return rows


def _build_split_summary(
    split_map: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "build_config": config,
        "splits": {
            split_name: _summarize_split(samples)
            for split_name, samples in split_map.items()
        },
        "rows": _build_split_summary_rows(split_map),
    }


def save_manifest(samples: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _save_manifest_bundle(
    split_map: Dict[str, List[Dict[str, Any]]],
    output_root: Union[str, Path],
    config: Dict[str, Any],
) -> Dict[str, str]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_paths: Dict[str, str] = {}
    for split_name, samples in split_map.items():
        manifest_path = output_root / f"{split_name}_manifest.jsonl"
        save_manifest(samples, manifest_path)
        manifest_paths[f"{split_name}_manifest"] = str(manifest_path)

    with open(output_root / "manifest_all.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                split_name: samples
                for split_name, samples in split_map.items()
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    split_summary = _build_split_summary(split_map=split_map, config=config)
    with open(output_root / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)

    manifest_paths["split_summary"] = str(output_root / "split_summary.json")
    manifest_paths["manifest_all"] = str(output_root / "manifest_all.json")
    manifest_paths["output_root"] = str(output_root)
    return manifest_paths


def _group_samples_by_condition(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["condition_uid"], []).append(sample)
    return grouped


def _build_condition_records(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_samples_by_condition(samples)
    records: List[Dict[str, Any]] = []

    for condition_uid, condition_samples in sorted(grouped.items()):
        label_set = {sample["label_3c"] for sample in condition_samples}
        if len(label_set) != 1:
            raise ValueError(
                f"condition {condition_uid} 存在多标签，无法做 grouped-stratified 划分: {sorted(label_set)}"
            )

        sample0 = condition_samples[0]
        records.append(
            {
                "condition_uid": condition_uid,
                "condition_id": sample0["condition_id"],
                "dataset_domain": sample0["dataset_domain"],
                "role": sample0["role"],
                "label_3c": sample0["label_3c"],
                "label_2c": sample0["label_2c"],
                "n_samples": len(condition_samples),
            }
        )

    return records


def _choose_condition_split_counts(
    n_conditions: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int]:
    if n_conditions < 3 and (val_ratio > 0 or test_ratio > 0):
        raise ValueError(f"condition 数过少，无法切分 train/val/test: {n_conditions}")

    n_val = int(round(n_conditions * val_ratio)) if val_ratio > 0 else 0
    n_test = int(round(n_conditions * test_ratio)) if test_ratio > 0 else 0

    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if test_ratio > 0 and n_test == 0:
        n_test = 1

    while n_val + n_test >= n_conditions:
        if n_test > n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    if n_val + n_test >= n_conditions:
        raise ValueError(
            f"condition 数不足以留出训练集: n_conditions={n_conditions}, "
            f"n_val={n_val}, n_test={n_test}"
        )

    return n_val, n_test


def _split_condition_records_train_val_test(
    condition_records: List[Dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in CLASS_NAMES_3}

    for record in condition_records:
        by_label[record["label_3c"]].append(record)

    missing_labels = [label for label, items in by_label.items() if not items]
    if missing_labels:
        raise ValueError(f"grouped split 缺少类别 condition: {missing_labels}")

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for label in CLASS_NAMES_3:
        label_records = list(by_label[label])
        rng.shuffle(label_records)

        n_val, n_test = _choose_condition_split_counts(
            n_conditions=len(label_records),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        val_records.extend(label_records[:n_val])
        test_records.extend(label_records[n_val:n_val + n_test])
        train_records.extend(label_records[n_val + n_test:])

    return train_records, val_records, test_records


def _assign_condition_records_to_folds(
    condition_records: List[Dict[str, Any]],
    num_folds: int,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    if num_folds < 2:
        raise ValueError(f"num_folds 必须 >= 2，当前: {num_folds}")

    rng = random.Random(seed)
    folds: List[List[Dict[str, Any]]] = [[] for _ in range(num_folds)]
    by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in CLASS_NAMES_3}

    for record in condition_records:
        by_label[record["label_3c"]].append(record)

    for label in CLASS_NAMES_3:
        label_records = list(by_label[label])
        if len(label_records) < num_folds:
            raise ValueError(
                f"类别 {label} 的 condition 数 ({len(label_records)}) 少于 num_folds={num_folds}，"
                "无法构建每折都有该类的 grouped CV"
            )

        rng.shuffle(label_records)
        label_records.sort(key=lambda item: (-item["n_samples"], item["condition_uid"]))

        label_fold_condition_counts = [0] * num_folds
        for record in label_records:
            best_fold = min(
                range(num_folds),
                key=lambda idx: (
                    label_fold_condition_counts[idx],
                    sum(item["n_samples"] for item in folds[idx]),
                    len(folds[idx]),
                    idx,
                ),
            )
            folds[best_fold].append(record)
            label_fold_condition_counts[best_fold] += 1

    return folds


def _split_fold_train_val_condition_records(
    train_pool_records: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in CLASS_NAMES_3}

    for record in train_pool_records:
        by_label[record["label_3c"]].append(record)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    for label in CLASS_NAMES_3:
        label_records = list(by_label[label])
        if len(label_records) < 2:
            raise ValueError(
                f"fold 训练池中类别 {label} 的 condition 数不足以再切 val: {len(label_records)}"
            )

        rng.shuffle(label_records)
        n_val = int(round(len(label_records) * val_ratio)) if val_ratio > 0 else 0
        if val_ratio > 0 and n_val == 0:
            n_val = 1
        while n_val >= len(label_records):
            n_val -= 1

        val_records.extend(label_records[:n_val])
        train_records.extend(label_records[n_val:])

    return train_records, val_records


def _materialize_grouped_split_samples(
    samples: List[Dict[str, Any]],
    train_condition_uids: Sequence[str],
    val_condition_uids: Sequence[str],
    test_condition_uids: Sequence[str],
    balance_train: bool,
    balance_mode: str,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    train_set = set(train_condition_uids)
    val_set = set(val_condition_uids)
    test_set = set(test_condition_uids)

    source_train_raw = [
        _clone_sample_with_split(sample, "train")
        for sample in samples
        if sample["condition_uid"] in train_set
    ]
    source_val = [
        _clone_sample_with_split(sample, "val")
        for sample in samples
        if sample["condition_uid"] in val_set
    ]
    source_test = [
        _clone_sample_with_split(sample, "test")
        for sample in samples
        if sample["condition_uid"] in test_set
    ]

    if balance_train and balance_mode == "upsample_to_max":
        source_train = _balance_train_samples_upsample_to_max(source_train_raw, seed=seed)
    else:
        source_train = source_train_raw

    split_map = {
        "train": source_train,
        "val": source_val,
        "test": source_test,
    }

    for split_name, split_samples in split_map.items():
        _validate_split_integrity(
            split_name,
            split_samples,
            expected_domain="data1",
            require_full_3class=True,
        )

    return split_map


def build_data1_grouped_protocol(
    data1_dir: Union[str, Path],
    output_root: Union[str, Path] = "/home/gjw/code/SLM_data/processed_qwen_data1_protocol",
    official_num_folds: int = 3,
    official_val_ratio: float = 0.25,
    quick_val_ratio: float = 0.2,
    quick_test_ratio: float = 0.2,
    balance_train: bool = True,
    balance_mode: str = "upsample_to_max",
    seed: int = 42,
) -> Dict[str, str]:
    """
    构建 data1-only 正式协议:
    - quick_dev: grouped-stratified 单次 train/val/test
    - official_cv: condition-level grouped-stratified K-fold
    """
    if balance_mode not in {"upsample_to_max", "none"}:
        raise ValueError(f"不支持的 balance_mode: {balance_mode}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    samples = _load_samples_from_dataset_dir(data1_dir, dataset_domain="data1")
    condition_records = _build_condition_records(samples)

    quick_train_records, quick_val_records, quick_test_records = _split_condition_records_train_val_test(
        condition_records,
        val_ratio=quick_val_ratio,
        test_ratio=quick_test_ratio,
        seed=seed,
    )
    quick_split_map = _materialize_grouped_split_samples(
        samples=samples,
        train_condition_uids=[record["condition_uid"] for record in quick_train_records],
        val_condition_uids=[record["condition_uid"] for record in quick_val_records],
        test_condition_uids=[record["condition_uid"] for record in quick_test_records],
        balance_train=balance_train,
        balance_mode=balance_mode,
        seed=seed,
    )
    quick_bundle = _save_manifest_bundle(
        quick_split_map,
        output_root / "quick_dev",
        config={
            "protocol": "quick_dev",
            "data1_dir": str(data1_dir),
            "output_root": str(output_root / "quick_dev"),
            "quick_val_ratio": quick_val_ratio,
            "quick_test_ratio": quick_test_ratio,
            "balance_train": balance_train,
            "balance_mode": balance_mode,
            "seed": seed,
            "train_condition_uids": [record["condition_uid"] for record in quick_train_records],
            "val_condition_uids": [record["condition_uid"] for record in quick_val_records],
            "test_condition_uids": [record["condition_uid"] for record in quick_test_records],
        },
    )

    official_cv_root = output_root / "official_cv"
    fold_records = _assign_condition_records_to_folds(
        condition_records=condition_records,
        num_folds=official_num_folds,
        seed=seed,
    )

    fold_summaries: List[Dict[str, Any]] = []
    for fold_idx, test_records in enumerate(fold_records, start=1):
        test_condition_uids = [record["condition_uid"] for record in test_records]
        test_condition_uid_set = set(test_condition_uids)
        train_pool_records = [
            record for record in condition_records
            if record["condition_uid"] not in test_condition_uid_set
        ]
        train_records, val_records = _split_fold_train_val_condition_records(
            train_pool_records=train_pool_records,
            val_ratio=official_val_ratio,
            seed=seed + fold_idx,
        )

        split_map = _materialize_grouped_split_samples(
            samples=samples,
            train_condition_uids=[record["condition_uid"] for record in train_records],
            val_condition_uids=[record["condition_uid"] for record in val_records],
            test_condition_uids=test_condition_uids,
            balance_train=balance_train,
            balance_mode=balance_mode,
            seed=seed + fold_idx,
        )

        fold_dir = official_cv_root / f"fold_{fold_idx:02d}"
        fold_bundle = _save_manifest_bundle(
            split_map,
            fold_dir,
            config={
                "protocol": "official_cv",
                "fold_index": fold_idx,
                "data1_dir": str(data1_dir),
                "output_root": str(fold_dir),
                "official_num_folds": official_num_folds,
                "official_val_ratio": official_val_ratio,
                "balance_train": balance_train,
                "balance_mode": balance_mode,
                "seed": seed,
                "train_condition_uids": [record["condition_uid"] for record in train_records],
                "val_condition_uids": [record["condition_uid"] for record in val_records],
                "test_condition_uids": test_condition_uids,
            },
        )
        fold_summaries.append(
            {
                "fold_index": fold_idx,
                "output_root": fold_bundle["output_root"],
                "train_manifest": fold_bundle["train_manifest"],
                "val_manifest": fold_bundle["val_manifest"],
                "test_manifest": fold_bundle["test_manifest"],
                "split_summary": fold_bundle["split_summary"],
            }
        )

    protocol_summary = {
        "build_config": {
            "protocol": "data1_grouped_protocol",
            "data1_dir": str(data1_dir),
            "output_root": str(output_root),
            "official_num_folds": official_num_folds,
            "official_val_ratio": official_val_ratio,
            "quick_val_ratio": quick_val_ratio,
            "quick_test_ratio": quick_test_ratio,
            "balance_train": balance_train,
            "balance_mode": balance_mode,
            "seed": seed,
            "n_samples": len(samples),
            "n_conditions": len(condition_records),
        },
        "condition_records": condition_records,
        "quick_dev": {
            "output_root": quick_bundle["output_root"],
            "train_manifest": quick_bundle["train_manifest"],
            "val_manifest": quick_bundle["val_manifest"],
            "test_manifest": quick_bundle["test_manifest"],
            "split_summary": quick_bundle["split_summary"],
        },
        "official_cv": fold_summaries,
    }

    with open(output_root / "protocol_summary.json", "w", encoding="utf-8") as f:
        json.dump(protocol_summary, f, indent=2, ensure_ascii=False)

    return {
        "output_root": str(output_root),
        "quick_dev_root": quick_bundle["output_root"],
        "official_cv_root": str(official_cv_root),
        "protocol_summary": str(output_root / "protocol_summary.json"),
    }


def build_manifest_from_metadata(
    data1_dir: Union[str, Path],
    data2_dir: Union[str, Path],
    output_root: Union[str, Path] = "/home/gjw/code/SLM_data/processed_qwen_balanced",
    source_val_ratio: float = 0.15,
    source_test_ratio: float = 0.15,
    balance_source_train: bool = True,
    balance_mode: str = "upsample_to_max",
    seed: int = 42,
) -> Dict[str, str]:
    """
    从 metadata 文件构建 source(data1) / target(data2) 协议 manifest。

    返回:
        dict: 包含 train/val/test/transfer_test manifest 路径
    """
    if balance_mode not in {"upsample_to_max", "none"}:
        raise ValueError(f"不支持的 balance_mode: {balance_mode}")

    random.seed(seed)
    np.random.seed(seed)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_samples = _load_samples_from_dataset_dir(data1_dir, dataset_domain="data1")
    target_samples_base = _load_samples_from_dataset_dir(data2_dir, dataset_domain="data2")

    source_train_raw, source_val, source_test = _split_source_samples(
        source_samples,
        source_val_ratio=source_val_ratio,
        source_test_ratio=source_test_ratio,
        seed=seed,
    )

    if balance_source_train and balance_mode == "upsample_to_max":
        source_train = _balance_train_samples_upsample_to_max(source_train_raw, seed=seed)
    else:
        source_train = source_train_raw

    transfer_test = [_clone_sample_with_split(sample, "transfer_test") for sample in target_samples_base]

    split_map = {
        "train": source_train,
        "val": source_val,
        "test": source_test,
        "transfer_test": transfer_test,
    }

    _validate_split_integrity("train", source_train, expected_domain="data1", require_full_3class=True)
    _validate_split_integrity("val", source_val, expected_domain="data1", require_full_3class=True)
    _validate_split_integrity("test", source_test, expected_domain="data1", require_full_3class=True)
    _validate_split_integrity("transfer_test", transfer_test, expected_domain="data2", require_full_3class=True)

    logger.info(
        "划分结果: train=%d, val=%d, test=%d, transfer_test=%d",
        len(source_train),
        len(source_val),
        len(source_test),
        len(transfer_test),
    )

    bundle = _save_manifest_bundle(
        split_map=split_map,
        output_root=output_root,
        config={
            "data1_dir": str(data1_dir),
            "data2_dir": str(data2_dir),
            "output_root": str(output_root),
            "source_val_ratio": source_val_ratio,
            "source_test_ratio": source_test_ratio,
            "balance_source_train": balance_source_train,
            "balance_mode": balance_mode,
            "seed": seed,
            "source_train_raw_count": len(source_train_raw),
            "source_train_balanced_count": len(source_train),
        },
    )

    logger.info("Manifest 保存至: %s", output_root)

    return {
        "train_manifest": bundle["train_manifest"],
        "val_manifest": bundle["val_manifest"],
        "test_manifest": bundle["test_manifest"],
        "transfer_test_manifest": bundle["transfer_test_manifest"],
        "split_summary": bundle["split_summary"],
        "output_root": bundle["output_root"],
    }


def build_cli_parser() -> argparse.ArgumentParser:
    """统一 CLI：支持 balanced source/target 协议和 data1-only grouped protocol。"""
    parser = argparse.ArgumentParser(
        description="构建 Qwen-SLM manifest：balanced source/target 协议或 data1 grouped 官方协议",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 1: balanced source(data1) / target(data2) 协议
    python src/dataset_qwen2vl_slm.py --build \
        --data1_dir "/home/gjw/code/SLM_data/Causal_Image_Data" \
        --data2_dir "/home/gjw/code/SLM_data/Causal_Image_Data2" \
        --output_root "/home/gjw/code/SLM_data/processed_qwen_balanced" \
        --source_val_ratio 0.15 \
        --source_test_ratio 0.15 \
        --balance_source_train true \
        --balance_mode upsample_to_max \
        --seed 42

示例 2: data1-only grouped official protocol
    python src/dataset_qwen2vl_slm.py --build_data1_protocol \
        --data1_dir "/home/gjw/code/SLM_data/Causal_Image_Data" \
        --protocol_output_root "/home/gjw/code/SLM_data/processed_qwen_data1_protocol" \
        --official_num_folds 3 \
        --official_val_ratio 0.25 \
        --quick_val_ratio 0.2 \
        --quick_test_ratio 0.2 \
        --protocol_balance_train true \
        --protocol_balance_mode upsample_to_max \
        --seed 42
        """
    )

    parser.add_argument("--build", action="store_true",
                        help="构建 balanced source(data1)/target(data2) manifest")
    parser.add_argument("--build_data1_protocol", action="store_true",
                        help="构建 data1-only grouped quick_dev + official_cv 协议")
    parser.add_argument("--data1_dir", type=str, required=True,
                        help="source data1 (Causal_Image_Data) 目录路径")
    parser.add_argument("--data2_dir", type=str, default=None,
                        help="target data2 (Causal_Image_Data2) 目录路径")
    parser.add_argument("--output_root", type=str,
                        default="/home/gjw/code/SLM_data/processed_qwen_balanced",
                        help="balanced 协议输出目录路径")
    parser.add_argument("--source_val_ratio", type=float, default=0.15,
                        help="source 验证集比例")
    parser.add_argument("--source_test_ratio", type=float, default=0.15,
                        help="source 测试集比例")
    parser.add_argument("--balance_source_train", type=str2bool, default=True,
                        help="是否仅对 source train 做类别均衡")
    parser.add_argument("--balance_mode", type=str, default="upsample_to_max",
                        choices=["upsample_to_max", "none"],
                        help="source train 的均衡方式")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--protocol_output_root", type=str,
                        default="/home/gjw/code/SLM_data/processed_qwen_data1_protocol",
                        help="data1 grouped protocol 输出目录")
    parser.add_argument("--official_num_folds", type=int, default=3,
                        help="official CV 折数")
    parser.add_argument("--official_val_ratio", type=float, default=0.25,
                        help="每个 official fold 中 train_pool 切 val 的比例")
    parser.add_argument("--quick_val_ratio", type=float, default=0.2,
                        help="quick_dev grouped split 的 val 比例")
    parser.add_argument("--quick_test_ratio", type=float, default=0.2,
                        help="quick_dev grouped split 的 test 比例")
    parser.add_argument("--protocol_balance_train", type=str2bool, default=True,
                        help="是否对 protocol train 做类别均衡")
    parser.add_argument("--protocol_balance_mode", type=str, default="upsample_to_max",
                        choices=["upsample_to_max", "none"],
                        help="protocol train 的均衡方式")
    return parser


def main_build_manifest():
    """命令行入口。"""
    parser = build_cli_parser()

    args = parser.parse_args()

    if not args.build and not args.build_data1_protocol:
        parser.error("至少指定 --build 或 --build_data1_protocol 之一")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.build:
        if not args.data2_dir:
            parser.error("--build 模式需要提供 --data2_dir")

        result = build_manifest_from_metadata(
            data1_dir=args.data1_dir,
            data2_dir=args.data2_dir,
            output_root=args.output_root,
            source_val_ratio=args.source_val_ratio,
            source_test_ratio=args.source_test_ratio,
            balance_source_train=args.balance_source_train,
            balance_mode=args.balance_mode,
            seed=args.seed,
        )

        print("\n" + "="*60)
        print("✅ Balanced Manifest 构建完成")
        print("="*60)
        print(f"Train: {result['train_manifest']}")
        print(f"Val:   {result['val_manifest']}")
        print(f"Test:  {result['test_manifest']}")
        print(f"Transfer Test: {result['transfer_test_manifest']}")
        print(f"Summary: {result['split_summary']}")
        print("="*60)

    if args.build_data1_protocol:
        result = build_data1_grouped_protocol(
            data1_dir=args.data1_dir,
            output_root=args.protocol_output_root,
            official_num_folds=args.official_num_folds,
            official_val_ratio=args.official_val_ratio,
            quick_val_ratio=args.quick_val_ratio,
            quick_test_ratio=args.quick_test_ratio,
            balance_train=args.protocol_balance_train,
            balance_mode=args.protocol_balance_mode,
            seed=args.seed,
        )

        print("\n" + "="*60)
        print("✅ Data1 Grouped Protocol 构建完成")
        print("="*60)
        print(f"Output Root:     {result['output_root']}")
        print(f"Quick Dev Root:  {result['quick_dev_root']}")
        print(f"Official CV Root:{result['official_cv_root']}")
        print(f"Summary:         {result['protocol_summary']}")
        print("="*60)


# =============================================================================
# 入口判断
# =============================================================================

if __name__ == "__main__":
    # 如果带构建参数则执行 manifest 构建模式
    if any(arg in {"--build", "--build_data1_protocol"} for arg in sys.argv[1:]):
        main_build_manifest()
    else:
        # 自测模式
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        
        print("=" * 80)
        print("SLM 重构版数据集模块自测")
        print("=" * 80)
        
        print("\n三分类 token 映射:")
        for k in CLASS_NAMES_3:
            print(f"  {k} -> letters:{CLASS_TOKEN_MAP_3['letters'][k]} | words:{CLASS_TOKEN_MAP_3['words'][k]}")
        
        print("\n二分类 token 映射:")
        for k in CLASS_NAMES_2:
            print(f"  {k} -> letters:{CLASS_TOKEN_MAP_2['letters'][k]} | words:{CLASS_TOKEN_MAP_2['words'][k]}")
        
        demo_after = ["path_rgb1_after.jpg", "path_rgb2_after.jpg", "path_ir_after.jpg"]
        
        print("\n示例三分类对话:")
        conv = build_conversation(
            images_after=demo_after,
            prompt=choose_prompt("3class", "letters"),
            target_text="B",
        )
        print(json.dumps(conv, indent=2, ensure_ascii=False))
        
        print("\n示例二分类对话:")
        conv = build_conversation(
            images_after=demo_after,
            prompt=choose_prompt("2class", "letters"),
            target_text="Y",
        )
        print(json.dumps(conv, indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 80)
        print("提示: 使用 --build 参数运行 manifest 构建模式")
        print("  python src/dataset_qwen2vl_slm.py --build --help")
        print("=" * 80)
