import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.datasets import get_dataloaders
from src.settings import COMMON, VALIDATION, resolve_num_workers
from src.training.checkpoint import load_tinycnn_checkpoint
from src.training.engine import validate_one_epoch


def main():
    common = COMMON
    cfg = VALIDATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = resolve_num_workers(common.num_workers)

    _, val_loader, _ = get_dataloaders(
        data_root=common.data_root,
        batch_size=common.batch_size,
        num_workers=num_workers,
        val_split=common.val_split,
        image_size=common.image_size,
        normalize=True,
        apply_exposure=common.apply_exposure,
        cache_processed=common.cache_processed,
        load_train=False,
        load_val=True,
        load_test=False,
    )

    model, _ = load_tinycnn_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        num_classes=common.num_classes,
        device=device,
        enable_quant_stubs=False,
    )

    criterion = nn.CrossEntropyLoss()
    metrics = validate_one_epoch(model, val_loader, criterion, device)

    print(
        f"checkpoint={cfg.checkpoint_path} "
        f"val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.4f} "
        f"val_precision={metrics['precision']:.4f} val_recall={metrics['recall']:.4f} "
        f"val_f1={metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
