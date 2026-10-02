import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.datasets import get_dataloaders
from src.settings import COMMON, PYTORCH_TEST, ensure_parent_dir, resolve_num_workers
from src.training.checkpoint import load_tinycnn_checkpoint
from src.training.engine import predict


def main():
    common = COMMON
    cfg = PYTORCH_TEST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = resolve_num_workers(common.num_workers)

    _, _, test_loader = get_dataloaders(
        data_root=common.data_root,
        batch_size=common.batch_size,
        num_workers=num_workers,
        val_split=common.val_split,
        image_size=common.image_size,
        normalize=True,
        apply_exposure=common.apply_exposure,
        cache_processed=common.cache_processed,
        load_train=False,
        load_val=False,
        load_test=True,
    )

    model, _ = load_tinycnn_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        num_classes=common.num_classes,
        device=device,
        enable_quant_stubs=False,
    )

    ids, preds = predict(model, test_loader, device)
    ensure_parent_dir(cfg.output_csv)
    pd.DataFrame({"Id": ids, "Label": preds}).to_csv(cfg.output_csv, index=False)
    print(f"Saved: {cfg.output_csv}")


if __name__ == "__main__":
    main()
