"""Training, validation, best-checkpoint saving, and CLI runner for CV experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cv.config import CVConfig, DEFAULT_EXPERIMENTS_YAML, seed_everything
from src.cv.dataset import ThumbnailDataset, build_transforms, resolve_normalization
from src.cv.model import ThumbnailRegressor


class MultiTaskLoss(nn.Module):
    """Combined SmoothL1 (standardized `y`) + weighted BCEWithLogits (`y_top`) loss."""

    def __init__(
        self,
        cls_loss_weight: float = 1.0,
        reg_loss_beta: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss = nn.SmoothL1Loss(beta=reg_loss_beta)
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_y_target = (targets[:, 0] - y_mean) / y_std
        reg = self.reg_loss(preds[:, 0], norm_y_target)
        cls = self.cls_loss(preds[:, 1], targets[:, 1])
        total = reg + self.cls_loss_weight * cls
        return total, reg, cls


def _pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = float(np.sqrt((yt ** 2).sum() * (yp ** 2).sum()))
    return float((yt * yp).sum() / denom) if denom > 1e-12 else 0.0


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC using average-rank Mann-Whitney U without extra dependencies."""
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = float(ranks[pos].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _set_backbone_trainable(model: ThumbnailRegressor, trainable: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Run one training epoch and return the sample-weighted average total loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, targets, _, channel_idx, category_idx in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        channel_idx = channel_idx.to(device, non_blocking=True)
        category_idx = category_idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                preds = model(images, channel_idx=channel_idx, category_idx=category_idx)
                loss, _, _ = criterion(preds, targets, y_mean=y_mean, y_std=y_std)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images, channel_idx=channel_idx, category_idx=category_idx)
            loss, _, _ = criterion(preds, targets, y_mean=y_mean, y_std=y_std)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    y_mean: float = 0.0,
    y_std: float = 1.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the model on validation and return metrics plus unstandardized predictions."""
    model.eval()
    running_loss = 0.0
    running_reg = 0.0
    running_cls = 0.0
    correct_top = 0
    total_samples = 0

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_top_true: list[np.ndarray] = []
    all_top_prob: list[np.ndarray] = []
    records: list[dict[str, float | str]] = []

    for images, targets, video_ids, channel_idx, category_idx in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        channel_idx = channel_idx.to(device, non_blocking=True)
        category_idx = category_idx.to(device, non_blocking=True)

        preds = model(images, channel_idx=channel_idx, category_idx=category_idx)
        loss, reg_loss, cls_loss = criterion(preds, targets, y_mean=y_mean, y_std=y_std)

        probs_top = torch.sigmoid(preds[:, 1])
        pred_labels = (probs_top >= 0.5).float()
        correct_top += int((pred_labels == targets[:, 1]).sum().item())

        pred_y_unscaled = preds[:, 0] * y_std + y_mean

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_reg += reg_loss.item() * batch_size
        running_cls += cls_loss.item() * batch_size
        total_samples += batch_size

        y_true_np = targets[:, 0].cpu().numpy()
        y_pred_np = pred_y_unscaled.cpu().numpy()
        top_true_np = targets[:, 1].cpu().numpy()
        top_prob_np = probs_top.cpu().numpy()

        all_y_true.append(y_true_np)
        all_y_pred.append(y_pred_np)
        all_top_true.append(top_true_np)
        all_top_prob.append(top_prob_np)

        for vid, pred_y, pred_y_top in zip(video_ids, y_pred_np, top_prob_np):
            records.append(
                {
                    "video_id": vid,
                    "pred_y": float(pred_y),
                    "pred_y_top": float(pred_y_top),
                }
            )

    y_true_cat = np.concatenate(all_y_true) if all_y_true else np.array([])
    y_pred_cat = np.concatenate(all_y_pred) if all_y_pred else np.array([])
    top_true_cat = np.concatenate(all_top_true) if all_top_true else np.array([])
    top_prob_cat = np.concatenate(all_top_prob) if all_top_prob else np.array([])

    denom = max(total_samples, 1)
    metrics = {
        "val_loss": running_loss / denom,
        "val_reg_loss": running_reg / denom,
        "val_cls_loss": running_cls / denom,
        "val_top_acc": correct_top / denom,
        "val_mae": float(np.abs(y_true_cat - y_pred_cat).mean()) if len(y_true_cat) else 0.0,
        "val_pearson": _pearson_corr(y_true_cat, y_pred_cat),
        "val_auc": _roc_auc(top_true_cat, top_prob_cat),
    }
    preds_df = pd.DataFrame(records, columns=["video_id", "pred_y", "pred_y_top"])
    return metrics, preds_df


def run_training(config: CVConfig) -> tuple[Path, Path]:
    """Run the full training loop and save the best weights and validation CSV."""
    seed_everything(config.seed)
    device = torch.device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    norm_mean, norm_std = resolve_normalization(config.model_name)

    train_ds = ThumbnailDataset(
        splits_csv=config.splits_csv,
        images_dir=config.images_dir,
        split="train",
        transform=build_transforms(
            config.img_size,
            is_train=True,
            use_hflip=config.use_hflip,
            mean=norm_mean,
            std=norm_std,
        ),
        id_col=config.id_col,
        split_col=config.split_col,
        image_col=config.image_col,
        channel_col=config.channel_col,
        category_col=config.category_col,
        target_cols=config.target_cols,
    )
    # Reuse train vocabularies on validation to avoid data leakage.
    val_ds = ThumbnailDataset(
        splits_csv=config.splits_csv,
        images_dir=config.images_dir,
        split="val",
        transform=build_transforms(
            config.img_size,
            is_train=False,
            mean=norm_mean,
            std=norm_std,
        ),
        id_col=config.id_col,
        split_col=config.split_col,
        image_col=config.image_col,
        channel_col=config.channel_col,
        category_col=config.category_col,
        target_cols=config.target_cols,
        channel_vocab=train_ds.channel_vocab,
        category_vocab=train_ds.category_vocab,
    )

    if config.standardize_y and train_ds.has_targets and len(train_ds) > 0:
        y_vals = train_ds.df[config.target_cols[0]].to_numpy(dtype=np.float64)
        y_mean = float(y_vals.mean())
        y_std = max(float(y_vals.std()), 1e-6)
    else:
        y_mean, y_std = 0.0, 1.0

    pos_weight_tensor: Optional[torch.Tensor] = None
    if config.auto_pos_weight and train_ds.has_targets and len(train_ds) > 0:
        top_vals = train_ds.df[config.target_cols[1]].to_numpy(dtype=np.float64)
        n_pos = float((top_vals == 1).sum())
        n_neg = float((top_vals == 0).sum())
        if n_pos > 0 and n_neg > 0:
            pos_weight_tensor = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    model = ThumbnailRegressor(
        model_name=config.model_name,
        pretrained=config.pretrained,
        dropout=config.dropout,
        head_hidden_dim=config.head_hidden_dim,
        use_metadata=config.use_metadata,
        num_channels=len(train_ds.channel_vocab),
        num_categories=len(train_ds.category_vocab),
        channel_emb_dim=config.channel_emb_dim,
        category_emb_dim=config.category_emb_dim,
    ).to(device)

    criterion = MultiTaskLoss(
        cls_loss_weight=config.cls_loss_weight,
        reg_loss_beta=config.reg_loss_beta,
        pos_weight=pos_weight_tensor,
    )

    head_params = [
        p for name, p in model.named_parameters() if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.lr * config.backbone_lr_mult},
            {"params": head_params, "lr": config.lr},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    if config.freeze_epochs > 0:
        _set_backbone_trainable(model, False)

    best_val_loss = float("inf")
    best_weights_path = config.output_dir / "best_model.pt"
    best_preds_path = config.output_dir / "val_predictions.csv"

    for epoch in range(1, config.epochs + 1):
        if config.freeze_epochs > 0 and epoch == config.freeze_epochs + 1:
            _set_backbone_trainable(model, True)
            print(f"Epoch {epoch:02d}: Unfreezing backbone with lr={config.lr * config.backbone_lr_mult:.2e}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            y_mean=y_mean,
            y_std=y_std,
            scaler=scaler,
        )
        metrics, val_preds_df = validate(
            model, val_loader, criterion, device, y_mean=y_mean, y_std=y_std
        )
        scheduler.step()

        val_loss = metrics["val_loss"]
        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"mae={metrics['val_mae']:.4f} | r={metrics['val_pearson']:.3f} | "
            f"auc={metrics['val_auc']:.3f} | acc={metrics['val_top_acc']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_name": config.model_name,
                    "img_size": config.img_size,
                    "dropout": config.dropout,
                    "head_hidden_dim": config.head_hidden_dim,
                    "use_metadata": config.use_metadata,
                    "channel_vocab": train_ds.channel_vocab,
                    "category_vocab": train_ds.category_vocab,
                    "channel_emb_dim": config.channel_emb_dim,
                    "category_emb_dim": config.category_emb_dim,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "norm_mean": list(norm_mean),
                    "norm_std": list(norm_std),
                    "state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                },
                best_weights_path,
            )
            val_preds_df.to_csv(best_preds_path, index=False)

    print(
        f"Training complete. Best val_loss={best_val_loss:.4f} | "
        f"weights={best_weights_path} | predictions={best_preds_path}"
    )
    return best_weights_path, best_preds_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a YouTubePredictor CV experiment from configs/cv_experiments.yaml."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline_cnn",
        help="Experiment key in configs/cv_experiments.yaml (e.g., convnext_tiny, finetuning_siglip).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_EXPERIMENTS_YAML,
        help="Path to the experiments YAML file.",
    )
    parser.add_argument("--splits-csv", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CVConfig.from_experiment(
        experiment_name=args.experiment,
        yaml_path=args.config_path,
        splits_csv=args.splits_csv,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    run_training(config)


if __name__ == "__main__":
    main()
