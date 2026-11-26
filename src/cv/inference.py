"""Inference helpers exposed for the application."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
import torch
import torch.nn as nn

from src.cv.config import IMG_SIZE, MODEL_NAME, default_device
from src.cv.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
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
    target_device = torch.device(device or default_device())
    # weights_only=True restricts loading to tensor data and avoids arbitrary pickle execution.
    checkpoint = torch.load(weights_path, map_location=target_device, weights_only=True)

    # Support both the metadata dict saved by train.py and a raw state_dict.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        resolved_model_name = model_name or checkpoint.get("model_name", MODEL_NAME)
        resolved_img_size = img_size or int(checkpoint.get("img_size", IMG_SIZE))
        dropout = float(checkpoint.get("dropout", 0.2))
        head_hidden_dim = int(checkpoint.get("head_hidden_dim", 256))
        norm_mean = tuple(float(x) for x in checkpoint.get("norm_mean", ()))
        norm_std = tuple(float(x) for x in checkpoint.get("norm_std", ()))
        state_dict = checkpoint["state_dict"]
    else:
        resolved_model_name = model_name or MODEL_NAME
        resolved_img_size = img_size or IMG_SIZE
        dropout = 0.2
        head_hidden_dim = 256
        norm_mean = ()
        norm_std = ()
        state_dict = checkpoint

    if len(norm_mean) != 3 or len(norm_std) != 3:
        norm_mean, norm_std = resolve_normalization(resolved_model_name)

    # Skip downloading pretrained weights at inference time since we load our trained checkpoint.
    model = ThumbnailRegressor(
        model_name=resolved_model_name,
        pretrained=False,
        dropout=dropout,
        head_hidden_dim=head_hidden_dim,
    )
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    # Attach normalization stats so predict_thumbnail uses the exact training preprocessing.
    model.norm_mean = norm_mean  # type: ignore[attr-defined]
    model.norm_std = norm_std  # type: ignore[attr-defined]
    return model, resolved_img_size, target_device


@torch.no_grad()
def predict_thumbnail(
    image: str | Path | Image.Image,
    model: Optional[nn.Module] = None,
    weights_path: Optional[str | Path] = None,
    img_size: int = IMG_SIZE,
    device: Optional[str | torch.device] = None,
) -> dict[str, float]:
    """Predict `pred_y` (regression) and `pred_y_top` (top-quartile probability).

    Accepts a file path or an already opened `PIL.Image.Image` (useful when
    handling uploads in Streamlit or Gradio).
    """
    if model is None:
        if weights_path is None:
            raise ValueError("Provide either a loaded `model` or `weights_path`.")
        model, img_size, target_device = load_model(
            weights_path=weights_path,
            img_size=img_size,
            device=str(device) if device is not None else None,
        )
    else:
        target_device = torch.device(
            device if device is not None else next(model.parameters()).device
        )
        model.eval()

    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            pil_image = img.convert("RGB")
    else:
        pil_image = image.convert("RGB")

    norm_mean = getattr(model, "norm_mean", IMAGENET_MEAN)
    norm_std = getattr(model, "norm_std", IMAGENET_STD)
    transform = build_transforms(
        img_size=img_size, is_train=False, mean=norm_mean, std=norm_std
    )
    tensor = transform(pil_image).unsqueeze(0).to(target_device)

    raw_out = model(tensor).squeeze(0)
    pred_y = float(raw_out[0].cpu().item())
    pred_y_top = float(torch.sigmoid(raw_out[1]).cpu().item())
    return {
        "pred_y": pred_y,
        "pred_y_top": pred_y_top,
    }


def predict(
    image_path: str | Path,
    weights_path: str | Path,
    device: Optional[str] = None,
) -> dict[str, float]:
    """Convenience wrapper for single-image inference from disk paths."""
    return predict_thumbnail(
        image=image_path,
        weights_path=weights_path,
        device=device,
    )
