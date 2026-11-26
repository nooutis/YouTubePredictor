"""Training, validation, best-checkpoint saving, and CLI runner for CV experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cv.config import CVConfig, DEFAULT_EXPERIMENTS_YAML, seed_everything
from src.cv.dataset import ThumbnailDataset, build_transforms, resolve_normalization
from src.cv.model import ThumbnailRegressor


class MultiTaskLoss(nn.Module):
    """Combined SmoothL1 (regression for `y`) + BCEWithLogits (binary `y_top`) loss."""

    def __init__(self, cls_loss_weight: float = 1.0, reg_loss_beta: float = 1.0) -> None:
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss = nn.SmoothL1Loss(beta=reg_loss_beta)
        self.cls_loss = nn.BCEWithLogitsLoss()

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reg = self.reg_loss(preds[:, 0], targets[:, 0])
        cls = self.cls_loss(preds[:, 1], targets[:, 1])
        total = reg + self.cls_loss_weight * cls
        return total, reg, cls


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Run one training epoch and return the sample-weighted average total loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                preds = model(images)
                loss, _, _ = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss, _, _ = criterion(preds, targets)
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
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the model on the validation set and return metrics plus predictions."""
    model.eval()
    running_loss = 0.0
    running_reg = 0.0
    running_cls = 0.0
    correct_top = 0
    total_samples = 0
    records: list[dict[str, float | str]] = []

    for images, targets, video_ids in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(images)
        loss, reg_loss, cls_loss = criterion(preds, targets)

        probs_top = torch.sigmoid(preds[:, 1])
        pred_labels = (probs_top >= 0.5).float()
        correct_top += int((pred_labels == targets[:, 1]).sum().item())

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_reg += reg_loss.item() * batch_size
        running_cls += cls_loss.item() * batch_size
        total_samples += batch_size

        pred_y_np = preds[:, 0].cpu().numpy()
        prob_top_np = probs_top.cpu().numpy()
        for vid, pred_y, pred_y_top in zip(video_ids, pred_y_np, prob_top_np):
            records.append(
                {
                    "video_id": vid,
                    "pred_y": float(pred_y),
                    "pred_y_top": float(pred_y_top),
                }
            )

    denom = max(total_samples, 1)
    metrics = {
        "val_loss": running_loss / denom,
        "val_reg_loss": running_reg / denom,
        "val_cls_loss": running_cls / denom,
        "val_top_acc": correct_top / denom,
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
        target_cols=config.target_cols,
    )
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
        target_cols=config.target_cols,
    )

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
    ).to(device)

    criterion = MultiTaskLoss(
        cls_loss_weight=config.cls_loss_weight,
        reg_loss_beta=config.reg_loss_beta,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_weights_path = config.output_dir / "best_model.pt"
    best_preds_path = config.output_dir / "val_predictions.csv"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
        metrics, val_preds_df = validate(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = metrics["val_loss"]
        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_reg={metrics['val_reg_loss']:.4f} | "
            f"val_cls={metrics['val_cls_loss']:.4f} | "
            f"val_top_acc={metrics['val_top_acc']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_name": config.model_name,
                    "img_size": config.img_size,
                    "dropout": config.dropout,
                    "head_hidden_dim": config.head_hidden_dim,
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
