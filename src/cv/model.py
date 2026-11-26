"""Dual-branch multi-task architecture (regression + top-quartile classification) for thumbnails."""

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
    """Pretrained vision backbone with separate MLP heads for `y` and `y_top`.

    Returns a tensor of shape `(batch_size, 2)` where index 0 is the continuous
    regression prediction for `y` and index 1 is the raw logit for `y_top`.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: bool = True,
        dropout: float = 0.2,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.dropout = dropout
        self.head_hidden_dim = head_hidden_dim

        # Setting num_classes=0 removes the classifier and returns pooled features directly.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        in_features = self.backbone.num_features

        self.head_reg = _build_mlp_branch(in_features, head_hidden_dim, dropout)
        self.head_cls = _build_mlp_branch(in_features, head_hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pred_y = self.head_reg(features)
        logit_y_top = self.head_cls(features)
        return torch.cat([pred_y, logit_y_top], dim=1)
