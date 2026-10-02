import os
import sys

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.datasets import auto_exposure_balance
from src.settings import COMMON, TFLITE_FLOAT_TEST, ensure_parent_dir

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def preprocess_image(path: str, expects_nhwc: bool, apply_exposure: bool) -> np.ndarray:
    image = cv2.imread(path)
    if apply_exposure:
        image = auto_exposure_balance(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if not expects_nhwc:
        image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)


def main():
    common = COMMON
    cfg = TFLITE_FLOAT_TEST
    test_dir = "test"
    image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(VALID_EXTENSIONS)])

    interpreter = tf.lite.Interpreter(model_path=cfg.model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    expects_nhwc = bool(input_details["shape"][-1] == 3)

    ids, preds = [], []
    for file_name in image_files:
        path = os.path.join(test_dir, file_name)
        x = preprocess_image(path, expects_nhwc=expects_nhwc, apply_exposure=common.apply_exposure)
        interpreter.set_tensor(input_details["index"], x.astype(input_details["dtype"]))
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details["index"])
        ids.append(os.path.splitext(file_name)[0])
        preds.append(int(np.argmax(logits, axis=1)[0]))

    ensure_parent_dir(cfg.output_csv)
    pd.DataFrame({"Id": ids, "Label": preds}).to_csv(cfg.output_csv, index=False)
    print(f"Saved: {cfg.output_csv}")


if __name__ == "__main__":
    main()
