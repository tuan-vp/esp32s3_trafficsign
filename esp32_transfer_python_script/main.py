import requests
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ESP32_IP = "192.168.1.14"
BASE_URL = f"http://{ESP32_IP}"
IMG_SIZE = 32

TEST_DIR = "offline_test"
OUTPUT_CSV = "result.csv"


def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8).tobytes()


def infer_one(image_path):
    raw = preprocess(image_path)

    resp = requests.post(
        f"{BASE_URL}/infer",
        data=raw,
        headers={"Content-Type": "application/octet-stream"},
        timeout=10,
    )

    if resp.status_code != 200:
        raise RuntimeError(resp.text)

    return resp.json()


def main():
    results = []

    for img_path in sorted(Path(TEST_DIR).glob("*.png")):
        img_id = img_path.stem.zfill(5)

        try:
            r = infer_one(str(img_path))

            pred = r["class"]
            latency = r["infer_ms"]

            results.append({
                "Id": img_id,
                "Label": pred,
                "Latency(ms)": latency
            })

            print(f"{img_id} | pred={pred} | {latency:.1f} ms")

        except Exception as e:
            print(f"{img_id} | ERROR: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
