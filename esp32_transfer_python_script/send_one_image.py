import requests
import cv2
import numpy as np
import sys
import time

ESP32_IP = "192.168.1.14"
BASE_URL = f"http://{ESP32_IP}"
IMG_SIZE = 32

def preprocess(image_path: str) -> bytes:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được: {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8).tobytes()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test.png"

    raw = preprocess(path)

    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/infer",
        data=raw,
        headers={"Content-Type": "application/octet-stream"},
        timeout=15,
    )
    rtt_ms = (time.perf_counter() - t0) * 1000

    r = resp.json()
    print(f"File      : {path}")
    print(f"Class     : {r['class']}")
    print(f"Score     : {r['score']} / 255")
    print(f"Scores    : {r['scores']}")
    print(f"Infer ESP : {r['infer_ms']:.2f} ms")
    print(f"RTT       : {rtt_ms:.1f} ms")

if __name__ == "__main__":
    main()
