"""Multimodal dual-branch architecture (thumbnail + metadata) for `y` and `y_top`."""

from typing import Optional

import timm
import torch
import torch.nn as nn

from src.cv.config import MODEL_NAME


def _build_mlp_branch(in_features: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Build a single-output MLP branch with LayerNorm, GELU, and Dropout."""
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, 1),
    )


class ThumbnailRegressor(nn.Module):
    """Pretrained vision backbone fused with optional channel/category embeddings.

    Returns a tensor of shape `(batch_size, 2)` where index 0 is the continuous
    regression prediction for `y` and index 1 is the raw logit for `y_top`.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: bool = True,
        dropout: float = 0.2,
        head_hidden_dim: int = 256,
        use_metadata: bool = True,
        num_channels: int = 1,
        num_categories: int = 1,
        channel_emb_dim: int = 32,
        category_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.dropout = dropout
        self.head_hidden_dim = head_hidden_dim
        self.use_metadata = use_metadata
        self.channel_emb_dim = channel_emb_dim
        self.category_emb_dim = category_emb_dim

        # Setting num_classes=0 removes the classifier and returns pooled features directly.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        in_features = self.backbone.num_features

        if use_metadata:
            self.channel_emb = nn.Embedding(max(num_channels, 1), channel_emb_dim)
            self.category_emb = nn.Embedding(max(num_categories, 1), category_emb_dim)
            in_features += channel_emb_dim + category_emb_dim
        else:
            self.channel_emb = None
            self.category_emb = None

        self.head_reg = _build_mlp_branch(in_features, head_hidden_dim, dropout)
        self.head_cls = _build_mlp_branch(in_features, head_hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        channel_idx: Optional[torch.Tensor] = None,
        category_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.backbone(x)

        if self.use_metadata and self.channel_emb is not None and self.category_emb is not None:
            batch_size = x.size(0)
            if channel_idx is None:
                channel_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            if category_idx is None:
                category_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            features = torch.cat(
                [features, self.channel_emb(channel_idx), self.category_emb(category_idx)],
                dim=1,
            )

        pred_y = self.head_reg(features)
        logit_y_top = self.head_cls(features)
        return torch.cat([pred_y, logit_y_top], dim=1)
