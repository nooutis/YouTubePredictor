"""Inference helpers exposed for the application."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
import torch.nn as nn

from src.cv.config import IMG_SIZE, MODEL_NAME, default_device, resolve_torch_device
from src.cv.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    UNK_TOKEN,
    build_transforms,
    resolve_normalization,
)
from src.cv.model import ThumbnailRegressor


def load_model(
    weights_path: str | Path,
    model_name: Optional[str] = None,
    img_size: Optional[int] = None,
    device: Optional[str] = None,
) -> tuple[nn.Module, int, torch.device]:
    """Load a checkpoint saved by `run_training` and set the model to eval mode."""
    target_device, _ = resolve_torch_device(device or default_device())
    # Always map tensors to CPU first so XLA/CUDA checkpoints load reliably everywhere.
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        resolved_model_name = model_name or checkpoint.get("model_name", MODEL_NAME)
        resolved_img_size = img_size or int(checkpoint.get("img_size", IMG_SIZE))
        dropout = float(checkpoint.get("dropout", 0.2))
        drop_path_rate = float(checkpoint.get("drop_path_rate", 0.0))
        head_hidden_dim = int(checkpoint.get("head_hidden_dim", 256))
        use_metadata = bool(checkpoint.get("use_metadata", False))
        use_channel_emb = bool(
            checkpoint.get("use_channel_emb", "channel_emb.weight" in state_dict)
        )
        channel_vocab = dict(checkpoint.get("channel_vocab", {UNK_TOKEN: 0}))
        category_vocab = dict(checkpoint.get("category_vocab", {UNK_TOKEN: 0}))
        channel_emb_dim = int(checkpoint.get("channel_emb_dim", 32))
        category_emb_dim = int(checkpoint.get("category_emb_dim", 16))
        tabular_cols = list(checkpoint.get("tabular_cols", []))
        tabular_stats = checkpoint.get("tabular_stats")
        tabular_hidden_dim = int(checkpoint.get("tabular_hidden_dim", 64))
        use_letterbox = bool(checkpoint.get("use_letterbox", True))
        y_mean = float(checkpoint.get("y_mean", 0.0))
        y_std = float(checkpoint.get("y_std", 1.0))
        norm_mean = tuple(float(x) for x in checkpoint.get("norm_mean", ()))
        norm_std = tuple(float(x) for x in checkpoint.get("norm_std", ()))
    else:
        state_dict = checkpoint
        resolved_model_name = model_name or MODEL_NAME
        resolved_img_size = img_size or IMG_SIZE
        dropout = 0.2
        drop_path_rate = 0.0
        head_hidden_dim = 256
        use_metadata = False
        use_channel_emb = "channel_emb.weight" in state_dict
        channel_vocab = {UNK_TOKEN: 0}
        category_vocab = {UNK_TOKEN: 0}
        channel_emb_dim = 32
        category_emb_dim = 16
        tabular_cols = []
        tabular_stats = None
        tabular_hidden_dim = 64
        use_letterbox = True
        y_mean = 0.0
        y_std = 1.0
        norm_mean = ()
        norm_std = ()

    if len(norm_mean) != 3 or len(norm_std) != 3:
        norm_mean, norm_std = resolve_normalization(resolved_model_name)

    model = ThumbnailRegressor(
        model_name=resolved_model_name,
        pretrained=False,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        head_hidden_dim=head_hidden_dim,
        use_metadata=use_metadata,
        use_channel_emb=use_channel_emb,
        num_channels=len(channel_vocab),
        num_categories=len(category_vocab),
        channel_emb_dim=channel_emb_dim,
        category_emb_dim=category_emb_dim,
        num_tabular_features=len(tabular_cols),
        tabular_hidden_dim=tabular_hidden_dim,
    )
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    model.norm_mean = norm_mean  # type: ignore[attr-defined]
    model.norm_std = norm_std  # type: ignore[attr-defined]
    model.channel_vocab = channel_vocab  # type: ignore[attr-defined]
    model.category_vocab = category_vocab  # type: ignore[attr-defined]
    model.tabular_cols = tabular_cols  # type: ignore[attr-defined]
    model.tabular_stats = tabular_stats  # type: ignore[attr-defined]
    model.use_letterbox = use_letterbox  # type: ignore[attr-defined]
    model.y_mean = y_mean  # type: ignore[attr-defined]
    model.y_std = y_std  # type: ignore[attr-defined]
    return model, resolved_img_size, target_device


def load_ensemble(
    weights_paths: list[str | Path],
    device: Optional[str] = None,
) -> list[tuple[nn.Module, int, torch.device]]:
    """Pre-load multiple fold checkpoints once for fast repeated ensemble inference."""
    if not weights_paths:
        raise ValueError("Provide at least one checkpoint path in `weights_paths`.")
    return [load_model(wp, device=device) for wp in weights_paths]


def _build_tabular_tensor(
    model: nn.Module,
    tabular_features: Optional[dict[str, float]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build a `(1, num_tabular_features)` tensor with train-median fallback."""
    tabular_cols = getattr(model, "tabular_cols", [])
    if not tabular_cols:
        return None
    stats = getattr(model, "tabular_stats", None) or {}
    medians = stats.get("median", [0.0] * len(tabular_cols))
    means = stats.get("mean", [0.0] * len(tabular_cols))
    stds = stats.get("std", [1.0] * len(tabular_cols))

    feats_in = tabular_features or {}
    row: list[float] = []
    for i, col in enumerate(tabular_cols):
        raw = feats_in.get(col)
        if raw is None or not math.isfinite(float(raw)):
            val = float(medians[i])
        else:
            val = float(raw)
        std_val = max(float(stds[i]), 1e-6)
        row.append((val - float(means[i])) / std_val)
    return torch.tensor([row], dtype=torch.float32, device=device)


@torch.no_grad()
def predict_thumbnail(
    image: str | Path | Image.Image,
    model: Optional[nn.Module] = None,
    weights_path: Optional[str | Path] = None,
    img_size: int = IMG_SIZE,
    device: Optional[str | torch.device] = None,
    channel_title: Optional[str] = None,
    category: Optional[str] = None,
    tabular_features: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Predict `pred_y` (regression) and `pred_y_top` (top-quartile probability)."""
    if model is None:
        if weights_path is None:
            raise ValueError("Provide either a loaded `model` or `weights_path`.")
        model, img_size, target_device = load_model(
            weights_path=weights_path,
            img_size=img_size,
            device=str(device) if device is not None else None,
        )
    else:
        if device is None:
            target_device = next(model.parameters()).device
        elif isinstance(device, torch.device):
            target_device = device
        else:
            target_device, _ = resolve_torch_device(device)
        model.eval()

    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            pil_image = img.convert("RGB")
    else:
        pil_image = image.convert("RGB")

    norm_mean = getattr(model, "norm_mean", IMAGENET_MEAN)
    norm_std = getattr(model, "norm_std", IMAGENET_STD)
    channel_vocab = getattr(model, "channel_vocab", {UNK_TOKEN: 0})
    category_vocab = getattr(model, "category_vocab", {UNK_TOKEN: 0})
    use_letterbox = bool(getattr(model, "use_letterbox", True))
    y_mean = float(getattr(model, "y_mean", 0.0))
    y_std = float(getattr(model, "y_std", 1.0))

    transform = build_transforms(
        img_size=img_size,
        is_train=False,
        mean=norm_mean,
        std=norm_std,
        use_letterbox=use_letterbox,
    )
    tensor = transform(pil_image).unsqueeze(0).to(target_device)

    ch_id = channel_vocab.get(str(channel_title), 0) if channel_title is not None else 0
    cat_id = category_vocab.get(str(category), 0) if category is not None else 0
    channel_idx = torch.tensor([ch_id], dtype=torch.long, device=target_device)
    category_idx = torch.tensor([cat_id], dtype=torch.long, device=target_device)
    tabular_tensor = _build_tabular_tensor(model, tabular_features, target_device)

    raw_out = model(
        tensor,
        channel_idx=channel_idx,
        category_idx=category_idx,
        tabular=tabular_tensor,
    ).squeeze(0)
    pred_y = float(raw_out[0].cpu().item()) * y_std + y_mean
    pred_y_top = float(torch.sigmoid(raw_out[1]).cpu().item())
    return {
        "pred_y": pred_y,
        "pred_y_top": pred_y_top,
    }


def predict(
    image_path: str | Path,
    weights_path: str | Path,
    device: Optional[str] = None,
    channel_title: Optional[str] = None,
    category: Optional[str] = None,
    tabular_features: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Convenience wrapper for single-image inference from disk paths."""
    return predict_thumbnail(
        image=image_path,
        weights_path=weights_path,
        device=device,
        channel_title=channel_title,
        category=category,
        tabular_features=tabular_features,
    )


def predict_ensemble(
    image: str | Path | Image.Image,
    weights_paths: Optional[list[str | Path]] = None,
    models: Optional[list[tuple[nn.Module, int, torch.device]]] = None,
    weights: Optional[list[float]] = None,
    threshold: float = 0.5,
    device: Optional[str] = None,
    channel_title: Optional[str] = None,
    category: Optional[str] = None,
    tabular_features: Optional[dict[str, float]] = None,
) -> dict[str, float | int]:
    """Weighted average of `pred_y` and `pred_y_top` across multiple fold models."""
    if models is None:
        if not weights_paths:
            raise ValueError("Provide either `models` or `weights_paths`.")
        loaded = load_ensemble(weights_paths, device=device)
    else:
        loaded = models

    if not loaded:
        raise ValueError("Ensemble must contain at least one model.")

    if weights is None:
        norm_weights = [1.0 / len(loaded)] * len(loaded)
    else:
        if len(weights) != len(loaded):
            raise ValueError("`weights` length must match number of ensemble models.")
        w_sum = sum(weights)
        if w_sum <= 0:
            raise ValueError("Sum of `weights` must be positive.")
        norm_weights = [float(w) / w_sum for w in weights]

    preds = [
        predict_thumbnail(
            image=image,
            model=m,
            img_size=sz,
            device=dev,
            channel_title=channel_title,
            category=category,
            tabular_features=tabular_features,
        )
        for (m, sz, dev) in loaded
    ]
    avg_y = sum(w * p["pred_y"] for w, p in zip(norm_weights, preds))
    avg_top = sum(w * p["pred_y_top"] for w, p in zip(norm_weights, preds))
    return {
        "pred_y": avg_y,
        "pred_y_top": avg_top,
        "pred_y_top_label": int(avg_top >= threshold),
    }
