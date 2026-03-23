from __future__ import annotations

import gc
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": torch.float16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_qwen_visual_module(
    model_name_or_path: str,
    dtype_name: str = "float16",
) -> Tuple[nn.Module, int, int]:
    dtype = resolve_dtype(dtype_name)
    full_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        device_map="cpu",
    )
    visual_model = full_model.model.visual
    hidden_size = int(full_model.config.vision_config.hidden_size)
    spatial_merge_size = int(full_model.config.vision_config.spatial_merge_size)
    del full_model
    gc.collect()
    return visual_model, hidden_size, spatial_merge_size


def configure_visual_trainability(
    visual_model: nn.Module,
    unfreeze_last_n_blocks: int,
) -> int:
    for param in visual_model.parameters():
        param.requires_grad = False

    if unfreeze_last_n_blocks <= 0:
        return 0

    blocks = list(visual_model.blocks)
    to_unfreeze = blocks[-unfreeze_last_n_blocks:]
    for block in to_unfreeze:
        block.float()
        for param in block.parameters():
            param.requires_grad = True

    visual_model.merger.float()
    for param in visual_model.merger.parameters():
        param.requires_grad = True

    return len(to_unfreeze)


class MLPFusionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_views: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        fused_dim = input_dim * num_views
        self.pre_norm = nn.LayerNorm(fused_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, image_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_input = image_vectors.reshape(image_vectors.shape[0], -1).to(dtype=self.pre_norm.weight.dtype)
        fused_repr = self.mlp(self.pre_norm(fused_input))
        logits = self.classifier(fused_repr)
        return logits, F.normalize(fused_repr, dim=-1)


class QwenVisionFusionClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_views: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        model_dtype: str = "float16",
        unfreeze_last_n_blocks: int = 0,
    ) -> None:
        super().__init__()
        visual_model, vision_hidden_size, spatial_merge_size = load_qwen_visual_module(
            model_name_or_path=model_name_or_path,
            dtype_name=model_dtype,
        )
        self.visual = visual_model
        self.vision_hidden_size = vision_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.num_views = num_views
        self.visual_dtype = resolve_dtype(model_dtype)
        self.num_trainable_visual_blocks = configure_visual_trainability(
            visual_model=self.visual,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        self.fusion_head = MLPFusionHead(
            input_dim=vision_hidden_size,
            num_views=num_views,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    @property
    def has_trainable_visual(self) -> bool:
        return any(param.requires_grad for param in self.visual.parameters())

    def forward_from_image_vectors(
        self,
        image_vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fusion_head(image_vectors)

    def extract_image_vectors(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        sample_image_counts: List[int],
    ) -> torch.Tensor:
        vision_context = nullcontext() if self.has_trainable_visual else torch.no_grad()
        with vision_context:
            autocast_context = (
                torch.autocast(device_type="cuda", enabled=(pixel_values.device.type == "cuda"), dtype=torch.float16)
                if pixel_values.device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                vision_outputs = self.visual(
                    pixel_values.to(dtype=self.visual_dtype),
                    grid_thw=image_grid_thw,
                )
        if isinstance(vision_outputs, torch.Tensor):
            pooled_tokens = vision_outputs
        elif hasattr(vision_outputs, "pooler_output"):
            pooled_tokens = vision_outputs.pooler_output
        elif hasattr(vision_outputs, "last_hidden_state"):
            pooled_tokens = vision_outputs.last_hidden_state
        else:
            raise TypeError(
                "Unsupported Qwen visual output type: "
                f"{type(vision_outputs)}. Expected Tensor or ModelOutput-like object."
            )
        split_sizes = (image_grid_thw.prod(-1) // (self.spatial_merge_size ** 2)).tolist()
        image_embeds = torch.split(pooled_tokens, split_sizes, dim=0)
        image_vectors = torch.stack([embed.mean(dim=0) for embed in image_embeds], dim=0)
        sample_chunks = torch.split(image_vectors, sample_image_counts, dim=0)
        return torch.stack(list(sample_chunks), dim=0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        sample_image_counts: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_vectors = self.extract_image_vectors(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            sample_image_counts=sample_image_counts,
        )
        return self.forward_from_image_vectors(image_vectors)
