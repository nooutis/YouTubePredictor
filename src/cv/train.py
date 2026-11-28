"""Training, validation, best-checkpoint saving, and CLI runner for CV experiments."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cv.config import (
    CVConfig,
    DEFAULT_EXPERIMENTS_YAML,
    resolve_torch_device,
    seed_everything,
)
from src.cv.dataset import (
    ThumbnailDataset,
    build_transforms,
    build_vocab,
    resolve_normalization,
)
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


def _set_backbone_trainable(
    model: ThumbnailRegressor,
    trainable: bool,
    unfreeze_last_n_blocks: int = 0,
) -> None:
    """Freeze or partially/fully unfreeze the vision backbone."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    if not trainable:
        return

    if unfreeze_last_n_blocks <= 0:
        for param in model.backbone.parameters():
            param.requires_grad = True
        return

    blocks = getattr(model.backbone, "blocks", None)
    stages = getattr(model.backbone, "stages", None)
    if blocks is not None and hasattr(blocks, "__len__"):
        n_blocks = min(unfreeze_last_n_blocks, len(blocks))
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for attr in ("norm", "attn_pool", "fc_norm", "head_drop", "head"):
            mod = getattr(model.backbone, attr, None)
            if mod is not None and isinstance(mod, nn.Module):
                for param in mod.parameters():
                    param.requires_grad = True
        return

    if stages is not None and hasattr(stages, "__len__"):
        n_stages = min(unfreeze_last_n_blocks, len(stages))
        for stage in stages[-n_stages:]:
            for param in stage.parameters():
                param.requires_grad = True
        for attr in ("norm_pre", "head"):
            mod = getattr(model.backbone, attr, None)
            if mod is not None and isinstance(mod, nn.Module):
                for param in mod.parameters():
                    param.requires_grad = True
        return

    for param in model.backbone.parameters():
        param.requires_grad = True


def _is_better_metric(
    candidate: float, best: float, selection_metric: str
) -> bool:
    """Compare validation metrics according to metric direction."""
    if selection_metric == "val_pearson":
        return candidate > best
    return candidate < best


def _resolve_features_csv(config: CVConfig) -> Optional[Path]:
    """Locate `features.csv` from config or alongside `splits_csv`."""
    if not config.use_tabular:
        return None
    if config.features_csv is not None and config.features_csv.is_file():
        return config.features_csv
    sibling = config.splits_csv.parent / "features.csv"
    if sibling.is_file():
        return sibling
    return None


def _cosine_warmup_lambda(
    epoch_idx: int, total_epochs: int, delay_epochs: int, warmup_epochs: int
) -> float:
    """Compute LR multiplier with optional freeze delay, linear warmup, and cosine decay."""
    if epoch_idx < delay_epochs:
        return 0.0
    active_epoch = epoch_idx - delay_epochs
    active_total = max(total_epochs - delay_epochs, 1)
    if warmup_epochs > 0 and active_epoch < warmup_epochs:
        return float(active_epoch + 1) / float(warmup_epochs)
    decay_total = max(active_total - warmup_epochs, 1)
    progress = min(max(active_epoch - warmup_epochs, 0) / float(decay_total), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    xm: object | None = None,
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 1,
    log_interval: Optional[int] = None,
) -> float:
    """Run one training epoch and return the sample-weighted average total loss."""
    model.train()
    running_loss_t = torch.zeros((), dtype=torch.float32, device=device)
    total_samples = 0
    num_batches = max(len(loader), 1)
    accum = max(int(grad_accum_steps), 1)
    step_log = log_interval or max(1, num_batches // 4)
    t0 = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets, _, channel_idx, category_idx, tabular) in enumerate(
        loader, start=1
    ):
        if xm is None:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            channel_idx = channel_idx.to(device, non_blocking=True)
            category_idx = category_idx.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)

        should_step = (step % accum == 0) or (step == num_batches)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                preds = model(
                    images,
                    channel_idx=channel_idx,
                    category_idx=category_idx,
                    tabular=tabular,
                )
                loss, _, _ = criterion(preds, targets, y_mean=y_mean, y_std=y_std)
                scaled_loss = loss / float(accum)
            scaler.scale(scaled_loss).backward()
            if should_step:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            preds = model(
                images,
                channel_idx=channel_idx,
                category_idx=category_idx,
                tabular=tabular,
            )
            loss, _, _ = criterion(preds, targets, y_mean=y_mean, y_std=y_std)
            (loss / float(accum)).backward()
            if should_step:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if xm is not None:
                    xm.optimizer_step(optimizer, barrier=True)  # type: ignore[attr-defined]
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        running_loss_t = running_loss_t + loss.detach() * batch_size
        total_samples += batch_size

        # Sync loss to host only at log intervals to avoid per-step XLA graph stalls.
        if step % step_log == 0 or step == num_batches:
            elapsed = time.perf_counter() - t0
            avg_so_far = float(running_loss_t.item()) / max(total_samples, 1)
            print(
                f"  [train] batch {step:>4d}/{num_batches} | "
                f"loss={avg_so_far:.4f} | {elapsed:.1f}s"
            )

    return float(running_loss_t.item()) / max(total_samples, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    xm: object | None = None,
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

    for images, targets, video_ids, channel_idx, category_idx, tabular in loader:
        if xm is None:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            channel_idx = channel_idx.to(device, non_blocking=True)
            category_idx = category_idx.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)

        preds = model(
            images,
            channel_idx=channel_idx,
            category_idx=category_idx,
            tabular=tabular,
        )
        loss, reg_loss, cls_loss = criterion(preds, targets, y_mean=y_mean, y_std=y_std)
        if xm is not None:
            xm.mark_step()  # type: ignore[attr-defined]

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
    mae = float(np.abs(y_true_cat - y_pred_cat).mean()) if len(y_true_cat) else 0.0
    rmse = (
        float(np.sqrt(np.mean((y_true_cat - y_pred_cat) ** 2)))
        if len(y_true_cat)
        else 0.0
    )
    metrics = {
        "val_loss": running_loss / denom,
        "val_reg_loss": running_reg / denom,
        "val_cls_loss": running_cls / denom,
        "val_top_acc": correct_top / denom,
        "val_mae": mae,
        "val_rmse": rmse,
        "val_pearson": _pearson_corr(y_true_cat, y_pred_cat),
        "val_auc": _roc_auc(top_true_cat, top_prob_cat),
    }
    preds_df = pd.DataFrame(records, columns=["video_id", "pred_y", "pred_y_top"])
    return metrics, preds_df


def run_training(config: CVConfig) -> tuple[Path, Path]:
    """Run the full training loop and save the best weights and validation CSV."""
    seed_everything(config.seed)
    device, xm = resolve_torch_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    norm_mean, norm_std = resolve_normalization(config.model_name)
    features_csv = _resolve_features_csv(config)

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
            use_letterbox=config.use_letterbox,
        ),
        id_col=config.id_col,
        split_col=config.split_col,
        image_col=config.image_col,
        channel_col=config.channel_col,
        category_col=config.category_col,
        target_cols=config.target_cols,
        features_csv=features_csv,
    )
    # Apply min_freq filtering to channel_vocab so rare channels train the <UNK> embedding.
    if config.channel_col in train_ds.df.columns and config.channel_min_freq > 1:
        train_ds.channel_vocab = build_vocab(
            train_ds.df[config.channel_col], min_freq=config.channel_min_freq
        )

    # Reuse train vocabularies and tabular stats on validation to avoid data leakage.
    val_ds = ThumbnailDataset(
        splits_csv=config.splits_csv,
        images_dir=config.images_dir,
        split="val",
        transform=build_transforms(
            config.img_size,
            is_train=False,
            mean=norm_mean,
            std=norm_std,
            use_letterbox=config.use_letterbox,
        ),
        id_col=config.id_col,
        split_col=config.split_col,
        image_col=config.image_col,
        channel_col=config.channel_col,
        category_col=config.category_col,
        target_cols=config.target_cols,
        channel_vocab=train_ds.channel_vocab,
        category_vocab=train_ds.category_vocab,
        features_csv=features_csv,
        tabular_cols=train_ds.tabular_cols,
        tabular_stats=train_ds.tabular_stats,
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
    train_loader: DataLoader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader: DataLoader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    if xm is not None:
        try:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader  # type: ignore[import-not-found]

            train_loader = MpDeviceLoader(train_loader, device)
            val_loader = MpDeviceLoader(val_loader, device)
        except Exception:
            pass

    model = ThumbnailRegressor(
        model_name=config.model_name,
        pretrained=config.pretrained,
        dropout=config.dropout,
        drop_path_rate=config.drop_path_rate,
        head_hidden_dim=config.head_hidden_dim,
        use_metadata=config.use_metadata,
        use_channel_emb=config.use_channel_emb,
        num_channels=len(train_ds.channel_vocab),
        num_categories=len(train_ds.category_vocab),
        channel_emb_dim=config.channel_emb_dim,
        category_emb_dim=config.category_emb_dim,
        num_tabular_features=len(train_ds.tabular_cols),
        tabular_hidden_dim=config.tabular_hidden_dim,
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda ep: _cosine_warmup_lambda(
                ep, config.epochs, config.freeze_epochs, config.warmup_epochs
            ),
            lambda ep: _cosine_warmup_lambda(ep, config.epochs, 0, config.warmup_epochs),
        ],
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    if config.freeze_epochs > 0:
        _set_backbone_trainable(model, False)

    pos_w_val = float(pos_weight_tensor.item()) if pos_weight_tensor is not None else 1.0
    print(
        f"[Config] model={config.model_name} | device={device} (xla={xm is not None}) | "
        f"img={config.img_size} | letterbox={config.use_letterbox} | "
        f"bs={config.batch_size}x{config.grad_accum_steps} | epochs={config.epochs} | "
        f"train={len(train_ds)} | val={len(val_ds)} | use_ch={config.use_channel_emb} | "
        f"categories={len(train_ds.category_vocab)} | tabular={len(train_ds.tabular_cols)} | "
        f"y_mean={y_mean:.3f} y_std={y_std:.3f} | cls_w={config.cls_loss_weight} pos_w={pos_w_val:.2f} | "
        f"freeze={config.freeze_epochs} unfreeze_blocks={config.unfreeze_last_n_blocks} | "
        f"select={config.selection_metric} patience={config.early_stopping_patience}"
    )

    selection_metric = config.selection_metric
    best_score = (
        float("-inf") if selection_metric == "val_pearson" else float("inf")
    )
    epochs_no_improve = 0
    best_weights_path = config.output_dir / "best_model.pt"
    best_preds_path = config.output_dir / "val_predictions.csv"

    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.perf_counter()
        if config.freeze_epochs > 0 and epoch == config.freeze_epochs + 1:
            _set_backbone_trainable(
                model,
                True,
                unfreeze_last_n_blocks=config.unfreeze_last_n_blocks,
            )
            trainable_backbone = sum(
                p.numel() for p in model.backbone.parameters() if p.requires_grad
            )
            total_backbone = sum(p.numel() for p in model.backbone.parameters())
            print(
                f"Epoch {epoch:02d}: Unfreezing backbone "
                f"({trainable_backbone:,}/{total_backbone:,} params, last_n={config.unfreeze_last_n_blocks}) "
                f"with backbone_lr={optimizer.param_groups[0]['lr']:.2e}, "
                f"head_lr={optimizer.param_groups[1]['lr']:.2e}"
            )

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            y_mean=y_mean,
            y_std=y_std,
            scaler=scaler,
            xm=xm,
            grad_clip_norm=config.grad_clip_norm,
            grad_accum_steps=config.grad_accum_steps,
        )
        metrics, val_preds_df = validate(
            model, val_loader, criterion, device, y_mean=y_mean, y_std=y_std, xm=xm
        )
        scheduler.step()

        current_score = metrics.get(selection_metric, metrics["val_mae"])
        is_best = _is_better_metric(current_score, best_score, selection_metric)
        epoch_dt = time.perf_counter() - epoch_t0
        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} ({epoch_dt:.1f}s) | "
            f"train_loss={train_loss:.4f} | val_loss={metrics['val_loss']:.4f} "
            f"(reg={metrics['val_reg_loss']:.4f}, cls={metrics['val_cls_loss']:.4f}) | "
            f"mae={metrics['val_mae']:.4f} | rmse={metrics['val_rmse']:.4f} | "
            f"r={metrics['val_pearson']:.3f} | auc={metrics['val_auc']:.3f} | "
            f"acc={metrics['val_top_acc']:.3f}"
            + (" | [*] BEST" if is_best else "")
        )

        if is_best:
            best_score = current_score
            epochs_no_improve = 0
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_name": config.model_name,
                    "img_size": config.img_size,
                    "dropout": config.dropout,
                    "drop_path_rate": config.drop_path_rate,
                    "head_hidden_dim": config.head_hidden_dim,
                    "use_metadata": config.use_metadata,
                    "use_channel_emb": config.use_channel_emb,
                    "channel_vocab": train_ds.channel_vocab,
                    "category_vocab": train_ds.category_vocab,
                    "channel_emb_dim": config.channel_emb_dim,
                    "category_emb_dim": config.category_emb_dim,
                    "tabular_cols": train_ds.tabular_cols,
                    "tabular_stats": train_ds.tabular_stats,
                    "tabular_hidden_dim": config.tabular_hidden_dim,
                    "use_letterbox": config.use_letterbox,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "norm_mean": list(norm_mean),
                    "norm_std": list(norm_std),
                    "state_dict": cpu_state,
                    "selection_metric": selection_metric,
                    "best_score": best_score,
                    "val_loss": metrics["val_loss"],
                    "val_mae": metrics["val_mae"],
                    "val_rmse": metrics["val_rmse"],
                    "val_pearson": metrics["val_pearson"],
                    "val_auc": metrics["val_auc"],
                    "epoch": epoch,
                },
                best_weights_path,
            )
            val_preds_df.to_csv(best_preds_path, index=False)
        else:
            epochs_no_improve += 1
            if (
                config.early_stopping_patience > 0
                and epochs_no_improve >= config.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch:02d} "
                    f"(no {selection_metric} improvement for {epochs_no_improve} epochs)."
                )
                break

    print(
        f"Training complete. Best {selection_metric}={best_score:.4f} | "
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
        help="Experiment key in configs/cv_experiments.yaml (e.g., convnext_tiny, finetuning_siglip_384).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_EXPERIMENTS_YAML,
        help="Path to the experiments YAML file.",
    )
    parser.add_argument("--splits-csv", type=Path, default=None)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--selection-metric", type=str, default=None)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CVConfig.from_experiment(
        experiment_name=args.experiment,
        yaml_path=args.config_path,
        splits_csv=args.splits_csv,
        features_csv=args.features_csv,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        selection_metric=args.selection_metric,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
    )
    run_training(config)


if __name__ == "__main__":
    main()
