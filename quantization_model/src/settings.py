from dataclasses import dataclass
import os
import random

import numpy as np
import torch


@dataclass(frozen=True)
class CommonConfig:
    data_root: str = "arguemted_data"
    batch_size: int = 32
    num_workers: int = -1
    val_split: float = 0.2
    image_size: int = 32
    num_classes: int = 10
    apply_exposure: bool = True
    cache_processed: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    min_epochs: int = 75
    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 2e-4
    label_smoothing: float = 0
    scheduler: str = "cosine"  # cosine | onecycle | none
    imbalance_strategy: str = "weighted_sampler"  # weighted_sampler | class_weight | none
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-4
    seed: int = 42
    checkpoint_path: str = "models/checkpoints/tinycnn_best.pth"
    final_path: str = "models/checkpoints/tinycnn_last.pth"


@dataclass(frozen=True)
class ValidationConfig:
    checkpoint_path: str = "models/checkpoints/tinycnn_best.pth"


@dataclass(frozen=True)
class PytorchTestConfig:
    checkpoint_path: str = "models/checkpoints/tinycnn_best.pth"
    output_csv: str = "results/pth_submission.csv"


@dataclass(frozen=True)
class PseudoLabelConfig:
    checkpoint_path: str = "models/checkpoints/tinycnn_best.pth"
    confidence_threshold: float = 0.60
    target_train_root: str = "arguemted_data"
    file_prefix: str = "pseudo"
    clear_old_pseudo: bool = True
    output_csv: str = "results/pseudo_labels.csv"


@dataclass(frozen=True)
class OnnxExportConfig:
    checkpoint_path: str = "models/checkpoints/tinycnn_best.pth"
    onnx_path: str = "models/exports/tinycnn.onnx"
    opset: int = 14


@dataclass(frozen=True)
class TFLiteExportConfig:
    onnx_path: str = "models/exports/tinycnn.onnx"
    output_dir: str = "models/exports/tflite_models"
    float_name: str = "tinycnn_float32.tflite"
    int8_name: str = "tinycnn_int8.tflite"
    calib_samples: int = 256


@dataclass(frozen=True)
class TFLiteInferenceConfig:
    model_path: str
    output_csv: str


COMMON = CommonConfig()
TRAIN = TrainConfig()
VALIDATION = ValidationConfig()
PYTORCH_TEST = PytorchTestConfig()
PSEUDO_LABEL = PseudoLabelConfig()
ONNX_EXPORT = OnnxExportConfig()
TFLITE_EXPORT = TFLiteExportConfig()
TFLITE_FLOAT_TEST = TFLiteInferenceConfig(
    model_path="models/exports/tflite_models/tinycnn_float32.tflite",
    output_csv="results/submission_tflite_float32.csv",
)
TFLITE_INT8_TEST = TFLiteInferenceConfig(
    model_path="models/exports/tflite_models/tinycnn_int8.tflite",
    output_csv="results/sample_submission.csv",
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_num_workers(requested: int, non_windows_cap: int = 8) -> int:
    if requested != -1:
        return requested
    cpu_count = os.cpu_count() or non_windows_cap
    return min(non_windows_cap, max(0, cpu_count // 2))


def ensure_parent_dir(path: str):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
