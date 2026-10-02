import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# =========================
# Config (easy to edit)
# =========================
INPUT_ROOT = Path("train")
OUTPUT_ROOT = Path("arguemted_data")
TARGET_IMAGES_PER_CLASS = 1500

REDUCE_X3_IF_GT = 1400
REDUCE_X2_MIN = 1000
REDUCE_X2_MAX = 1400

ROTATE_DEGREE = 10.0
DROPOUT_RATIO_IN_AUG = 0.4
DROPOUT_NUM_HOLES = (5, 8)
DROPOUT_HOLE_H = (10, 16)
DROPOUT_HOLE_W = (10, 16)
DROPOUT_PAD = 6
DROPOUT_NOISE = 15
RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def reduce_count_by_rule(total: int) -> int:
    if total > REDUCE_X3_IF_GT:
        return max(1, total // 3)
    if REDUCE_X2_MIN <= total <= REDUCE_X2_MAX:
        return max(1, total // 2)
    return total


def image_fingerprint(image_path: Path) -> bytes:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (image > image.mean()).astype(np.uint8)
    return np.packbits(bits).tobytes()


def select_diverse_subset(paths: list[Path], keep_count: int, rng: random.Random) -> list[Path]:
    if keep_count >= len(paths):
        return list(paths)

    groups: dict[bytes, list[Path]] = {}
    for path in paths:
        key = image_fingerprint(path)
        groups.setdefault(key, []).append(path)

    buckets = list(groups.values())
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)

    selected: list[Path] = []
    while len(selected) < keep_count:
        has_item = False
        for bucket in buckets:
            if bucket and len(selected) < keep_count:
                selected.append(bucket.pop())
                has_item = True
        if not has_item:
            break
    return selected


def rotate_small(image: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = image.shape[:2]
    angle = rng.uniform(-ROTATE_DEGREE, ROTATE_DEGREE)
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def coarse_dropout_similar_color(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    channels = 1 if out.ndim == 2 else out.shape[2]
    n_holes = rng.randint(DROPOUT_NUM_HOLES[0], DROPOUT_NUM_HOLES[1])

    for _ in range(n_holes):
        hole_h = min(h, rng.randint(DROPOUT_HOLE_H[0], DROPOUT_HOLE_H[1]))
        hole_w = min(w, rng.randint(DROPOUT_HOLE_W[0], DROPOUT_HOLE_W[1]))

        y1 = rng.randint(0, max(0, h - hole_h))
        x1 = rng.randint(0, max(0, w - hole_w))
        y2 = y1 + hole_h
        x2 = x1 + hole_w

        sy1 = max(0, y1 - DROPOUT_PAD)
        sx1 = max(0, x1 - DROPOUT_PAD)
        sy2 = min(h, y2 + DROPOUT_PAD)
        sx2 = min(w, x2 + DROPOUT_PAD)

        local_region = out[sy1:sy2, sx1:sx2]
        if local_region.size == 0:
            fill_color = np.full((channels,), 127, dtype=np.float32)
        else:
            if channels == 1:
                fill_color = np.array([float(local_region.mean())], dtype=np.float32)
            else:
                fill_color = local_region.reshape(-1, channels).mean(axis=0).astype(np.float32)

        patch_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        noise_shape = (hole_h, hole_w) if channels == 1 else (hole_h, hole_w, channels)
        noise = patch_rng.integers(-DROPOUT_NOISE, DROPOUT_NOISE + 1, size=noise_shape)

        patch = np.clip(fill_color + noise, 0, 255).astype(np.uint8)
        out[y1:y2, x1:x2] = patch

    return out


def augment_once(image: np.ndarray, apply_dropout: bool, rng: random.Random) -> np.ndarray:
    out = rotate_small(image, rng)
    if apply_dropout:
        out = coarse_dropout_similar_color(out, rng)
    return out


def build_class_dataset(class_dir: Path, out_dir: Path, rng: random.Random):
    image_paths = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
    total = len(image_paths)
    reduced = min(reduce_count_by_rule(total), TARGET_IMAGES_PER_CLASS)
    base_paths = select_diverse_subset(image_paths, reduced, rng)

    out_dir.mkdir(parents=True, exist_ok=True)

    # keep selected originals
    for idx, path in enumerate(base_paths):
        image = cv2.imread(str(path))
        cv2.imwrite(str(out_dir / f"{class_dir.name}_{idx:05d}.png"), image)

    # augment to target
    need = TARGET_IMAGES_PER_CLASS - len(base_paths)
    dropout_count = int(round(need * DROPOUT_RATIO_IN_AUG))
    dropout_flags = [True] * dropout_count + [False] * (need - dropout_count)
    rng.shuffle(dropout_flags)

    source_paths = base_paths if base_paths else image_paths
    for i, use_dropout in enumerate(dropout_flags):
        src_path = source_paths[i % len(source_paths)]
        src = cv2.imread(str(src_path))
        aug = augment_once(src, apply_dropout=use_dropout, rng=rng)
        out_index = len(base_paths) + i
        cv2.imwrite(str(out_dir / f"{class_dir.name}_{out_index:05d}.png"), aug)

    print(
        f"class={class_dir.name} "
        f"original={total} reduced={len(base_paths)} "
        f"augmented={need} dropout={dropout_count} "
        f"final={TARGET_IMAGES_PER_CLASS}"
    )


def main():
    rng = random.Random(RANDOM_SEED)
    class_dirs = sorted(
        [d for d in INPUT_ROOT.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name),
    )

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        build_class_dataset(class_dir=class_dir, out_dir=OUTPUT_ROOT / class_dir.name, rng=rng)

    print(f"\nDone. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
