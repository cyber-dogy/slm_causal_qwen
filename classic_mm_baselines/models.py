from __future__ import annotations

import torch
import torch.nn as nn

from classic_mm_baselines.encoders import MultiViewImageEncoder


class SingleViewFusion(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, view_index: int = 0) -> None:
        super().__init__()
        self.view_index = view_index
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(feats[:, self.view_index, :])


class MeanPoolFusion(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats.mean(dim=1))


class MLPConcatFusion(nn.Module):
    def __init__(self, in_dim: int, num_views: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(in_dim * num_views, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(feats.flatten(start_dim=1))
        return self.classifier(fused)


class GatedMLPFusion(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.gate = nn.Linear(in_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(feats)
        weights = torch.softmax(gate_logits, dim=1)
        fused = (feats * weights).sum(dim=1)
        fused = self.mlp(fused)
        return self.classifier(fused)


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(feats)
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        attended, _ = self.attn(query=cls, key=tokens, value=tokens, need_weights=False)
        x = self.norm1(cls + attended)
        x = self.norm2(x + self.ffn(x))
        return self.classifier(x[:, 0, :])


class LateFusion(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.per_view_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.gate = nn.Linear(in_dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        view_logits = self.per_view_head(feats)
        weights = torch.softmax(self.gate(feats), dim=1)
        return (view_logits * weights).sum(dim=1)


class MultiModalClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        fusion_type: str,
        num_classes: int,
        num_views: int,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = MultiViewImageEncoder(backbone_name=backbone_name, pretrained=pretrained)
        in_dim = self.encoder.feature_dim

        if fusion_type == "single_view":
            self.fusion = SingleViewFusion(in_dim=in_dim, num_classes=num_classes, view_index=0)
        elif fusion_type == "mean_pool":
            self.fusion = MeanPoolFusion(in_dim=in_dim, num_classes=num_classes, dropout=dropout)
        elif fusion_type == "mlp_concat":
            self.fusion = MLPConcatFusion(
                in_dim=in_dim,
                num_views=num_views,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif fusion_type == "gated_mlp":
            self.fusion = GatedMLPFusion(in_dim=in_dim, num_classes=num_classes, dropout=dropout)
        elif fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(in_dim=in_dim, num_classes=num_classes, dropout=dropout)
        elif fusion_type == "late_fusion":
            self.fusion = LateFusion(in_dim=in_dim, num_classes=num_classes, dropout=dropout)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)
        return self.fusion(feats)
