from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def macro_precision_recall_f1(all_labels: List[int], all_preds: List[int]) -> Tuple[float, float, float]:
    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)

    if y_true.size == 0:
        return 0.0, 0.0, 0.0

    num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def _epoch_pass(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()

        preds = torch.argmax(outputs, dim=1)
        batch_size = labels.size(0)

        running_loss += float(loss.item())
        total += batch_size
        correct += int((preds == labels).sum().item())

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    precision, recall, f1 = macro_precision_recall_f1(all_labels, all_preds)
    return {
        "loss": running_loss / max(len(loader), 1),
        "acc": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
) -> Dict[str, float]:
    return _epoch_pass(
        model,
        loader,
        criterion,
        device,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_per_batch=scheduler_step_per_batch,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    return _epoch_pass(model, loader, criterion, device, optimizer=None)


@torch.no_grad()
def predict(model: nn.Module, loader: Iterable, device: torch.device):
    model.eval()
    ids = []
    preds = []

    for images, file_names in loader:
        images = images.to(device)
        logits = model(images)
        batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
        ids.extend([name.rsplit(".", 1)[0] for name in file_names])
        preds.extend(batch_preds)

    return ids, preds
