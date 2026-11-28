"""Centralized hyperparameters and paths for the CV model.

Keeping these values in one place avoids editing multiple files when switching
between local runs and Kaggle, or when changing settings across experiments.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np
import torch
import yaml

# Defaults tuned for an initial baseline on a Kaggle GPU (T4/P100) or TPU.
IMG_SIZE: int = 224
BATCH_SIZE: int = 32
LR: float = 3e-4
EPOCHS: int = 15
SEED: int = 42
MODEL_NAME: str = "efficientnet_b0"
DEFAULT_EXPERIMENTS_YAML: Path = Path("configs/cv_experiments.yaml")


def default_device() -> str:
    """Pick the best available compute device: Kaggle TPU (XLA) -> CUDA -> MPS -> CPU."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

        _ = xm.xla_device()
        return "xla"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_device(device_str: str) -> tuple[torch.device, object | None]:
    """Resolve `xla`/`tpu`, `cuda`, `mps`, or `cpu` into a torch.device (and optional xm)."""
    if device_str.lower() in {"xla", "tpu"}:
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

            return xm.xla_device(), xm
        except Exception as exc:
            print(f"[Device] TPU/XLA requested but unavailable ({exc}); falling back.")
            fallback = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.device(fallback), None
    return torch.device(device_str), None


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
    splits_csv: Path = Path("/kaggle/working/YouTubePredictor/data/split.csv")
    features_csv: Optional[Path] = Path("/kaggle/working/YouTubePredictor/data/features.csv")
    images_dir: Path = Path(
        "/kaggle/input/datasets/charlottecarretta/youtube-datased-cleaned-with-images/thumbnails"
    )
    output_dir: Path = Path("/kaggle/working/outputs")

    # Image preprocessing and DataLoader settings.
    img_size: int = IMG_SIZE
    batch_size: int = BATCH_SIZE
    num_workers: int = 2
    use_hflip: bool = False
    use_letterbox: bool = False

    # Expected column names in splits.csv.
    id_col: str = "video_id"
    split_col: str = "split"
    image_col: str = "thumbnail_path"
    channel_col: str = "channel_title"
    category_col: str = "category"
    target_cols: tuple[str, str] = ("y", "y_top")

    # Backbone name from timm, multimodal metadata fusion, and two-branch MLP head.
    model_name: str = MODEL_NAME
    pretrained: bool = True
    dropout: float = 0.2
    drop_path_rate: float = 0.1
    head_hidden_dim: int = 256
    use_metadata: bool = True
    use_channel_emb: bool = False
    use_tabular: bool = True
    tabular_hidden_dim: int = 64
    channel_emb_dim: int = 32
    category_emb_dim: int = 16
    channel_min_freq: int = 3

    # Multi-task loss and target scaling settings (regression-primary).
    cls_loss_weight: float = 0.15
    reg_loss_beta: float = 1.0
    standardize_y: bool = True
    auto_pos_weight: bool = False

    # Fine-tuning schedule: freeze backbone first, then partial unfreeze with warmup + differential LR.
    freeze_epochs: int = 2
    warmup_epochs: int = 1
    backbone_lr_mult: float = 0.1
    unfreeze_last_n_blocks: int = 3
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 4
    selection_metric: str = "val_mae"

    # AdamW optimizer settings and training schedule length.
    lr: float = LR
    weight_decay: float = 0.05
    epochs: int = EPOCHS
    seed: int = SEED
    device: str = field(default_factory=default_device)

    def __post_init__(self) -> None:
        # Normalize string paths passed by callers into Path objects.
        self.splits_csv = Path(self.splits_csv)
        self.features_csv = Path(self.features_csv) if self.features_csv is not None else None
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
