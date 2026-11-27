"""Part 2 package for the YouTube thumbnail computer vision model.

Re-exports configuration, dataset, model, training, and inference utilities so
Kaggle notebooks and Persona 3's app can import them from `src.cv` directly.
"""

from src.cv.config import (
    BATCH_SIZE,
    EPOCHS,
    IMG_SIZE,
    LR,
    MODEL_NAME,
    SEED,
    CVConfig,
    default_device,
    resolve_torch_device,
    seed_everything,
)
from src.cv.dataset import ThumbnailDataset, build_transforms
from src.cv.inference import (
    load_ensemble,
    load_model,
    predict,
    predict_ensemble,
    predict_thumbnail,
)
from src.cv.model import ThumbnailRegressor
from src.cv.train import run_training, train_one_epoch, validate

__all__ = [
    "BATCH_SIZE",
    "CVConfig",
    "EPOCHS",
    "IMG_SIZE",
    "LR",
    "MODEL_NAME",
    "SEED",
    "ThumbnailDataset",
    "ThumbnailRegressor",
    "build_transforms",
    "default_device",
    "load_ensemble",
    "load_model",
    "predict",
    "predict_ensemble",
    "predict_thumbnail",
    "resolve_torch_device",
    "run_training",
    "seed_everything",
    "train_one_epoch",
    "validate",
]
