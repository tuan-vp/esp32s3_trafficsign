import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.datasets import get_dataloaders
from src.settings import COMMON, PSEUDO_LABEL, ensure_parent_dir, resolve_num_workers
from src.training.checkpoint import load_tinycnn_checkpoint


def clear_old_pseudo_files(train_root: Path, prefix: str):
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        for file_path in class_dir.glob(f"{prefix}_*"):
            if file_path.is_file():
                file_path.unlink()


def unique_destination(dst_dir: Path, stem: str, suffix: str) -> Path:
    candidate = dst_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = dst_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def main():
    common = COMMON
    cfg = PSEUDO_LABEL
    if not (0.0 <= cfg.confidence_threshold <= 1.0):
        raise ValueError("PSEUDO_LABEL.confidence_threshold must be in [0.0, 1.0]")

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

    test_dir = Path(test_loader.dataset.test_dir).resolve()
    train_root = Path(cfg.target_train_root).resolve()
    train_root.mkdir(parents=True, exist_ok=True)

    if cfg.clear_old_pseudo:
        clear_old_pseudo_files(train_root, cfg.file_prefix)

    selected_rows = []
    total_seen = 0
    total_selected = 0

    model.eval()
    with torch.no_grad():
        for images, file_names in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            confidences = confidences.cpu().tolist()
            preds = preds.cpu().tolist()
            total_seen += len(file_names)

            for file_name, pred_label, confidence in zip(file_names, preds, confidences):
                if confidence < cfg.confidence_threshold:
                    continue

                src_path = test_dir / file_name
                if not src_path.exists():
                    continue

                dst_dir = train_root / str(pred_label)
                dst_dir.mkdir(parents=True, exist_ok=True)

                src = Path(file_name)
                suffix = src.suffix if src.suffix else ".png"
                conf_tag = int(round(confidence * 10000.0))
                dst_stem = f"{cfg.file_prefix}_{src.stem}_c{pred_label}_p{conf_tag:04d}"
                dst_path = unique_destination(dst_dir, dst_stem, suffix)

                shutil.copy2(src_path, dst_path)
                total_selected += 1

                selected_rows.append(
                    {
                        "id": src.stem,
                        "file_name": file_name,
                        "pred_label": pred_label,
                        "probability": float(confidence),
                        "probability_pct": float(confidence * 100.0),
                        "src_path": str(src_path),
                        "dst_path": str(dst_path),
                    }
                )

    df = pd.DataFrame(selected_rows)
    if not df.empty:
        df = df.sort_values(["probability", "pred_label"], ascending=[False, True]).reset_index(drop=True)
        df["probability"] = df["probability"].round(6)
        df["probability_pct"] = df["probability_pct"].round(2)

    ensure_parent_dir(cfg.output_csv)
    df.to_csv(cfg.output_csv, index=False)

    print(
        f"Pseudo-label summary: selected={total_selected}/{total_seen} "
        f"(threshold={cfg.confidence_threshold:.2f})"
    )
    if not df.empty:
        per_class = df["pred_label"].value_counts().sort_index()
        for class_id, count in per_class.items():
            print(f"class={int(class_id)} added={int(count)}")
    print(f"Saved: {cfg.output_csv}")


if __name__ == "__main__":
    main()
