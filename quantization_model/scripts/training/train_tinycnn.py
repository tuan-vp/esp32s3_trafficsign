import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.datasets import get_dataloaders
from src.models.tinycnn import TinyCNN
from src.settings import COMMON, TRAIN, ensure_parent_dir, resolve_num_workers, set_seed
from src.training.engine import train_one_epoch, validate_one_epoch


def save_checkpoint(path: str, model: nn.Module, epoch: int, metrics: dict):
    ensure_parent_dir(path)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, path)


def build_sampler(train_subset, num_classes: int):
    labels = torch.tensor([train_subset.dataset.labels[i] for i in train_subset.indices], dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
    class_weights = labels.numel() / (num_classes * class_counts)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights.double(), num_samples=sample_weights.numel(), replacement=True)
    return sampler, class_weights


def main():
    common = COMMON
    cfg = TRAIN
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = resolve_num_workers(common.num_workers)

    train_loader, val_loader, _ = get_dataloaders(
        data_root=common.data_root,
        batch_size=common.batch_size,
        num_workers=num_workers,
        val_split=common.val_split,
        image_size=common.image_size,
        normalize=True,
        apply_exposure=common.apply_exposure,
        cache_processed=common.cache_processed,
        seed=cfg.seed,
        load_train=True,
        load_val=True,
        load_test=False,
    )

    class_weights = None
    if cfg.imbalance_strategy in {"weighted_sampler", "class_weight"}:
        sampler, class_weights = build_sampler(train_loader.dataset, common.num_classes)
        if cfg.imbalance_strategy == "weighted_sampler":
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=common.batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )

    model = TinyCNN(num_classes=common.num_classes).to(device)
    criterion_weight = class_weights.to(device) if cfg.imbalance_strategy == "class_weight" else None
    criterion = nn.CrossEntropyLoss(weight=criterion_weight, label_smoothing=cfg.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    scheduler = None
    scheduler_step_per_batch = False
    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    if cfg.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            epochs=cfg.epochs,
            steps_per_epoch=max(1, len(train_loader)),
            anneal_strategy="cos",
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=max(cfg.learning_rate / cfg.min_lr, 1.0),
        )
        scheduler_step_per_batch = True

    best_val_acc = -1.0
    best_epoch = 0
    wait_epochs = 0
    min_epochs = min(max(cfg.min_epochs, 1), cfg.epochs)

    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
        )
        val_metrics = validate_one_epoch(model=model, loader=val_loader, criterion=criterion, device=device)

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{cfg.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_recall={train_metrics['recall']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        improved = val_metrics["acc"] > (best_val_acc + cfg.early_stop_min_delta)
        if improved:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch + 1
            wait_epochs = 0
            save_checkpoint(cfg.checkpoint_path, model, epoch + 1, val_metrics)
        else:
            wait_epochs += 1

        if (epoch + 1) >= min_epochs and wait_epochs >= cfg.early_stop_patience:
            break

    save_checkpoint(cfg.final_path, model, best_epoch, {"best_val_acc": best_val_acc})
    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
