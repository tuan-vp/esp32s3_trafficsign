import os
import shutil
import sys
from importlib import metadata

import numpy as np
import onnx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.datasets import get_dataloaders
from src.settings import COMMON, TFLITE_EXPORT


if not hasattr(onnx, "mapping") and hasattr(onnx, "_mapping"):
    onnx.mapping = onnx._mapping


def _parse_version(version: str):
    parts = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _assert_tf_ml_dtypes_compat():
    try:
        tf_version = metadata.version("tensorflow")
        mld_version = metadata.version("ml-dtypes")
        np_version = metadata.version("numpy")
    except metadata.PackageNotFoundError:
        return

    tf_major, tf_minor, _ = _parse_version(tf_version)
    ml_major, ml_minor, _ = _parse_version(mld_version)
    np_major, _, _ = _parse_version(np_version)

    # TensorFlow 2.14/2.15 expects ml-dtypes 0.2.x in this pipeline.
    if (tf_major, tf_minor) <= (2, 15) and (ml_major, ml_minor) != (0, 2):
        raise RuntimeError(
            "Incompatible package versions detected for ONNX -> TFLite conversion:\n"
            f"- tensorflow=={tf_version}\n"
            f"- ml-dtypes=={mld_version}\n"
            f"- numpy=={np_version}\n"
            "Expected ml-dtypes 0.2.x with tensorflow <= 2.15.\n"
            "Fix: python -m pip install --force-reinstall \"ml-dtypes==0.2.0\" \"numpy==1.26.4\""
        )

    if (tf_major, tf_minor) <= (2, 15) and np_major >= 2:
        raise RuntimeError(
            "Incompatible package versions detected for ONNX -> TFLite conversion:\n"
            f"- tensorflow=={tf_version}\n"
            f"- numpy=={np_version}\n"
            "Expected numpy < 2.0 with tensorflow <= 2.15.\n"
            "Fix: python -m pip install --force-reinstall \"numpy==1.26.4\""
        )


_assert_tf_ml_dtypes_compat()
import tensorflow as tf
from onnx_tf.backend import prepare


def export_saved_model_from_onnx(onnx_path: str, saved_model_dir: str):
    model = onnx.load(onnx_path)
    prepare(model).export_graph(saved_model_dir)


def representative_dataset(train_loader, expects_nhwc: bool, max_samples: int):
    seen = 0
    for images, _ in train_loader:
        batch = images.numpy().astype(np.float32)
        if expects_nhwc:
            batch = np.transpose(batch, (0, 2, 3, 1))
        for i in range(batch.shape[0]):
            yield [np.expand_dims(batch[i], axis=0)]
            seen += 1
            if seen >= max_samples:
                return


def main():
    common = COMMON
    cfg = TFLITE_EXPORT
    # Keep calibration loader single-process for maximum portability on Windows.
    num_workers = 0
    os.makedirs(cfg.output_dir, exist_ok=True)
    float_path = os.path.join(cfg.output_dir, cfg.float_name)
    int8_path = os.path.join(cfg.output_dir, cfg.int8_name)

    tmp_dir = os.path.join(cfg.output_dir, "_onnx2tf_tmp")
    saved_model_dir = os.path.join(tmp_dir, "saved_model")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        export_saved_model_from_onnx(cfg.onnx_path, saved_model_dir)

        signature = tf.saved_model.load(saved_model_dir).signatures["serving_default"]
        input_tensor = next(iter(signature.structured_input_signature[1].values()))
        expects_nhwc = bool(input_tensor.shape[-1] == 3)

        train_loader, _, _ = get_dataloaders(
            data_root=common.data_root,
            batch_size=common.batch_size,
            num_workers=num_workers,
            val_split=common.val_split,
            image_size=common.image_size,
            normalize=True,
            apply_exposure=common.apply_exposure,
            cache_processed=common.cache_processed,
            load_train=True,
            load_val=False,
            load_test=False,
        )

        float_converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        float_tflite = float_converter.convert()
        with open(float_path, "wb") as f:
            f.write(float_tflite)

        int8_converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        int8_converter.representative_dataset = lambda: representative_dataset(
            train_loader, expects_nhwc=expects_nhwc, max_samples=max(1, cfg.calib_samples)
        )
        int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        int8_converter.inference_input_type = tf.int8
        int8_converter.inference_output_type = tf.int8
        int8_tflite = int8_converter.convert()
        with open(int8_path, "wb") as f:
            f.write(int8_tflite)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Saved float32: {float_path}")
    print(f"Saved int8: {int8_path}")


if __name__ == "__main__":
    main()
