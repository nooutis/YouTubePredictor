"""Centralized hyperparameters and paths for the CV model.

Keeping these values in one place avoids editing multiple files when switching
between local runs and Kaggle, or when changing settings across experiments.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml

# Defaults tuned for an initial baseline on a Kaggle GPU (T4/P100).
IMG_SIZE: int = 224
BATCH_SIZE: int = 32
LR: float = 3e-4
EPOCHS: int = 15
SEED: int = 42
MODEL_NAME: str = "efficientnet_b0"
DEFAULT_EXPERIMENTS_YAML: Path = Path("configs/cv_experiments.yaml")


def default_device() -> str:
    """Pick the best available compute device: CUDA first, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int = SEED) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch to make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Set the seed across all visible CUDA devices in multi-GPU setups.
        torch.cuda.manual_seed_all(seed)


@dataclass
class CVConfig:
    """Paths, CSV schema, model settings, and training hyperparameters."""

    # Default Kaggle paths; override these when running locally.
    splits_csv: Path = Path("/kaggle/input/youtube-predictor-data/splits.csv")
    images_dir: Path = Path("/kaggle/input/youtube-predictor-data/thumbnails")
    output_dir: Path = Path("/kaggle/working/outputs")

    # Image preprocessing and DataLoader settings.
    img_size: int = IMG_SIZE
    batch_size: int = BATCH_SIZE
    num_workers: int = 2
    use_hflip: bool = False

    # Expected column names in splits.csv (keep aligned with Persona 1).
    id_col: str = "video_id"
    split_col: str = "split"
    image_col: str = "thumbnail_path"
    target_cols: tuple[str, str] = ("y", "y_top")

    # Backbone name from timm and two-branch MLP head settings.
    model_name: str = MODEL_NAME
    pretrained: bool = True
    dropout: float = 0.2
    head_hidden_dim: int = 256

    # Multi-task loss settings: SmoothL1 for `y` + weighted BCEWithLogits for `y_top`.
    cls_loss_weight: float = 1.0
    reg_loss_beta: float = 1.0

    # AdamW optimizer settings and training schedule length.
    lr: float = LR
    weight_decay: float = 1e-2
    epochs: int = EPOCHS
    seed: int = SEED
    device: str = field(default_factory=default_device)

    def __post_init__(self) -> None:
        # Normalize string paths passed by callers into Path objects.
        self.splits_csv = Path(self.splits_csv)
        self.images_dir = Path(self.images_dir)
        self.output_dir = Path(self.output_dir)

    @classmethod
    def from_experiment(
        cls,
        experiment_name: str,
        yaml_path: str | Path = DEFAULT_EXPERIMENTS_YAML,
        **overrides: Any,
    ) -> "CVConfig":
        """Build a CVConfig from a named entry in `cv_experiments.yaml`."""
        with Path(yaml_path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        experiments = data.get("experiments", {})
        if experiment_name not in experiments:
            available = ", ".join(sorted(experiments.keys()))
            raise KeyError(
                f"Unknown experiment '{experiment_name}'. Available: {available}"
            )

        valid_fields = {f.name for f in fields(cls)}
        exp_kwargs = {
            k: v for k, v in experiments[experiment_name].items() if k in valid_fields
        }
        # Default output_dir per experiment unless explicitly overridden.
        exp_kwargs.setdefault("output_dir", Path("/kaggle/working/outputs") / experiment_name)
        for k, v in overrides.items():
            if v is not None and k in valid_fields:
                exp_kwargs[k] = v
        return cls(**exp_kwargs)
