"""PyTorch Dataset and image transforms for YouTube thumbnails."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image, ImageOps
import timm
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.cv.config import IMG_SIZE

# Standard ImageNet statistics used as fallback when model config is unavailable.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


class LetterboxPad:
    """Resize an image preserving aspect ratio and pad symmetrically to a square."""

    def __init__(
        self, size: int, fill: tuple[int, int, int] = (114, 114, 114)
    ) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.pad(
            img,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
            color=self.fill,
            centering=(0.5, 0.5),
        )


def resolve_normalization(
    model_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the `(mean, std)` expected by a `timm` backbone (e.g., SigLIP vs ImageNet)."""
    try:
        model = timm.create_model(model_name, pretrained=False)
        data_cfg = timm.data.resolve_model_data_config(model)
        mean = tuple(float(x) for x in data_cfg["mean"])
        std = tuple(float(x) for x in data_cfg["std"])
        if len(mean) == 3 and len(std) == 3:
            return mean, std  # type: ignore[return-value]
    except Exception:
        pass
    return IMAGENET_MEAN, IMAGENET_STD


def build_transforms(
    img_size: int = IMG_SIZE,
    is_train: bool = True,
    use_hflip: bool = False,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> Callable:
    """Build the torchvision transform pipeline for training or validation/inference.

    Preserves 16:9 thumbnail aspect ratio via letterbox padding and uses
    model-specific normalization statistics.
    """
    ops: list[Callable] = [LetterboxPad(img_size)]
    if is_train:
        if use_hflip:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.append(
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(ops)


class ThumbnailDataset(Dataset):
    """Dataset that reads `splits.csv` and loads thumbnail images with `y` and `y_top`."""

    def __init__(
        self,
        splits_csv: str | Path,
        images_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        id_col: str = "video_id",
        split_col: str = "split",
        image_col: str = "thumbnail_path",
        target_cols: tuple[str, str] = ("y", "y_top"),
    ) -> None:
        self.images_dir = Path(images_dir)
        self.split = split
        self.transform = transform or build_transforms(is_train=(split == "train"))
        self.id_col = id_col
        self.image_col = image_col
        self.target_cols = target_cols

        df = pd.read_csv(splits_csv)
        # Filter to the requested subset when a split column is present.
        if split_col in df.columns:
            df = df[df[split_col] == split].reset_index(drop=True)
        self.df = df

        # Test splits may omit target columns; in that case __getitem__ returns zeros.
        self.has_targets = all(col in self.df.columns for col in self.target_cols)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        """Locate the image file from the CSV path column or by trying common extensions."""
        # Check the explicit path column first (either absolute or relative to images_dir).
        if self.image_col in row and pd.notna(row[self.image_col]):
            candidate = Path(str(row[self.image_col]))
            if candidate.is_file():
                return candidate
            rel_candidate = self.images_dir / candidate
            if rel_candidate.is_file():
                return rel_candidate

        # Fall back to <video_id>.<ext> inside images_dir.
        video_id = str(row[self.id_col])
        for ext in IMAGE_EXTENSIONS:
            candidate = self.images_dir / f"{video_id}{ext}"
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"Thumbnail not found for video_id={video_id} in {self.images_dir}"
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        video_id = str(row[self.id_col])
        img_path = self._resolve_image_path(row)

        # Convert to RGB so PNGs with alpha channels or grayscale images do not break transforms.
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        tensor = self.transform(image)

        if self.has_targets:
            targets = torch.tensor(
                [float(row[self.target_cols[0]]), float(row[self.target_cols[1]])],
                dtype=torch.float32,
            )
        else:
            targets = torch.zeros(2, dtype=torch.float32)

        return tensor, targets, video_id
