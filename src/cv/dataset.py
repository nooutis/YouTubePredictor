"""PyTorch Dataset, metadata vocabulary helpers, and image transforms for YouTube thumbnails."""

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
UNK_TOKEN = "<UNK>"


def build_vocab(values: pd.Series, min_freq: int = 1) -> dict[str, int]:
    """Build a string-to-index mapping reserving index 0 for unknown/rare values."""
    vocab: dict[str, int] = {UNK_TOKEN: 0}
    counts = values.fillna(UNK_TOKEN).astype(str).value_counts()
    for val, freq in counts.items():
        if val != UNK_TOKEN and int(freq) >= min_freq and val not in vocab:
            vocab[val] = len(vocab)
    return vocab


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
    """Return the `(mean, std)` expected by a `timm` backbone without instantiating it."""
    try:
        cfg = timm.models.get_pretrained_cfg(model_name)
        if cfg is not None and cfg.mean and cfg.std:
            mean = tuple(float(x) for x in cfg.mean)
            std = tuple(float(x) for x in cfg.std)
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
    """Dataset that reads `splits.csv` and loads thumbnails, targets, and metadata."""

    def __init__(
        self,
        splits_csv: str | Path,
        images_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        id_col: str = "video_id",
        split_col: str = "split",
        image_col: str = "thumbnail_path",
        channel_col: str = "channel_title",
        category_col: str = "category",
        target_cols: tuple[str, str] = ("y", "y_top"),
        channel_vocab: Optional[dict[str, int]] = None,
        category_vocab: Optional[dict[str, int]] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.split = split
        self.transform = transform or build_transforms(is_train=(split == "train"))
        self.id_col = id_col
        self.image_col = image_col
        self.channel_col = channel_col
        self.category_col = category_col
        self.target_cols = target_cols

        df = pd.read_csv(splits_csv)
        if split_col in df.columns:
            df = df[df[split_col] == split].reset_index(drop=True)

        self.has_targets = all(col in df.columns for col in self.target_cols)
        if self.has_targets:
            # Drop any corrupted rows with non-numeric targets and coerce to float.
            y_num = pd.to_numeric(df[self.target_cols[0]], errors="coerce")
            y_top_num = pd.to_numeric(df[self.target_cols[1]], errors="coerce")
            valid_mask = y_num.notna() & y_top_num.notna()
            dropped = int((~valid_mask).sum())
            if dropped > 0:
                print(
                    f"[ThumbnailDataset:{split}] Dropped {dropped} rows with non-numeric targets."
                )
                df = df[valid_mask].reset_index(drop=True)
                y_num = y_num[valid_mask].reset_index(drop=True)
                y_top_num = y_top_num[valid_mask].reset_index(drop=True)
            df[self.target_cols[0]] = y_num.astype(float)
            df[self.target_cols[1]] = y_top_num.astype(float)

        self.df = df

        if channel_vocab is not None:
            self.channel_vocab = channel_vocab
        elif channel_col in self.df.columns:
            self.channel_vocab = build_vocab(self.df[channel_col])
        else:
            self.channel_vocab = {UNK_TOKEN: 0}

        if category_vocab is not None:
            self.category_vocab = category_vocab
        elif category_col in self.df.columns:
            self.category_vocab = build_vocab(self.df[category_col])
        else:
            self.category_vocab = {UNK_TOKEN: 0}

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        """Locate the image file from the CSV path column or by trying common extensions."""
        if self.image_col in row and pd.notna(row[self.image_col]):
            candidate = Path(str(row[self.image_col]))
            if candidate.is_file():
                return candidate
            rel_candidate = self.images_dir / candidate
            if rel_candidate.is_file():
                return rel_candidate

        video_id = str(row[self.id_col])
        for ext in IMAGE_EXTENSIONS:
            candidate = self.images_dir / f"{video_id}{ext}"
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"Thumbnail not found for video_id={video_id} in {self.images_dir}"
        )

    def _encode_meta(self, row: pd.Series, col: str, vocab: dict[str, int]) -> int:
        if col not in row or pd.isna(row[col]):
            return 0
        return vocab.get(str(row[col]), 0)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, int, int]:
        row = self.df.iloc[idx]
        video_id = str(row[self.id_col])
        img_path = self._resolve_image_path(row)

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

        channel_idx = self._encode_meta(row, self.channel_col, self.channel_vocab)
        category_idx = self._encode_meta(row, self.category_col, self.category_vocab)
        return tensor, targets, video_id, channel_idx, category_idx
