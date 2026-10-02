import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def auto_exposure_balance(
    img: np.ndarray,
    target_mean: int = 145,
    min_gamma: float = 0.6,
    max_gamma: float = 1.8,
    use_clahe: bool = True,
) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l = max(float(np.mean(l)), 1.0)
    gamma = np.log(target_mean / 255.0) / np.log(mean_l / 255.0)
    gamma = float(np.clip(gamma, min_gamma, max_gamma))
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)], dtype=np.uint8)
    l_adj = cv2.LUT(l, table)

    if use_clahe:
        l_adj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_adj)

    return cv2.cvtColor(cv2.merge([l_adj, a, b]), cv2.COLOR_LAB2BGR)


def _load_and_resize(path: str, size: int, apply_exposure: bool) -> np.ndarray:
    image = cv2.imread(path)
    if apply_exposure:
        image = auto_exposure_balance(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _looks_like_train_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    class_dirs = [d for d in os.listdir(path) if d.isdigit() and os.path.isdir(os.path.join(path, d))]
    return len(class_dirs) > 0


def _looks_like_test_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return any(file_name.lower().endswith(VALID_EXTENSIONS) for file_name in os.listdir(path))


def _resolve_train_dir(data_root: str) -> str:
    # Accept both:
    # 1) data_root points to dataset root containing /train
    # 2) data_root points directly to train directory
    nested_train = os.path.join(data_root, "train")
    if _looks_like_train_dir(nested_train):
        return nested_train
    if _looks_like_train_dir(data_root):
        return data_root
    raise FileNotFoundError(
        "Missing train directory. Expected one of:\n"
        f"- {nested_train}\n"
        f"- {data_root}\n"
        "Set COMMON.data_root correctly in src/settings.py."
    )


def _resolve_test_dir(data_root: str) -> str:
    # Accept both:
    # 1) data_root points to dataset root containing /test
    # 2) data_root points directly to test directory
    # 3) data_root points to train directory and sibling ../test exists
    candidates = [
        os.path.join(data_root, "test"),
        data_root,
        os.path.join(os.path.dirname(data_root), "test"),
    ]
    for path in candidates:
        if _looks_like_test_dir(path):
            return path
    raise FileNotFoundError(
        "Missing test directory. Expected one of:\n"
        + "\n".join(f"- {path}" for path in candidates)
        + "\nSet COMMON.data_root correctly in src/settings.py."
    )


class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        normalize: bool = True,
        size: int = 32,
        apply_exposure: bool = True,
        cache_processed: bool = True,
    ):
        self.root_dir = root_dir
        self.normalize = normalize
        self.size = size
        self.apply_exposure = apply_exposure
        self.cache_processed = cache_processed
        self.cache_dir = os.path.join(root_dir, "..", ".cache", f"train_s{size}_exp{int(apply_exposure)}")
        self.paths, self.labels = self._scan_train()

    def _scan_train(self):
        class_dirs = [d for d in os.listdir(self.root_dir) if d.isdigit()]
        class_dirs = sorted(class_dirs, key=lambda x: int(x))
        paths, labels = [], []
        for cls in class_dirs:
            cls_dir = os.path.join(self.root_dir, cls)
            for file_name in sorted(os.listdir(cls_dir)):
                if file_name.lower().endswith(VALID_EXTENSIONS):
                    paths.append(os.path.join(cls_dir, file_name))
                    labels.append(int(cls))
        return paths, labels

    def _cache_path(self, image_path: str) -> str:
        rel = os.path.relpath(image_path, self.root_dir)
        stem, _ = os.path.splitext(rel)
        return os.path.join(self.cache_dir, stem + ".npy")

    def _get_image(self, image_path: str) -> np.ndarray:
        if not self.cache_processed:
            return _load_and_resize(image_path, self.size, self.apply_exposure)

        cache_path = self._cache_path(image_path)
        if os.path.exists(cache_path):
            return np.load(cache_path)

        image = _load_and_resize(image_path, self.size, self.apply_exposure)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, image)
        return image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        image = self._get_image(self.paths[index])
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        if self.normalize:
            image = image / 255.0
        return image, self.labels[index]


class TestImageDataset(Dataset):
    def __init__(
        self,
        test_dir: str,
        normalize: bool = True,
        size: int = 32,
        apply_exposure: bool = True,
        cache_processed: bool = True,
    ):
        self.test_dir = test_dir
        self.normalize = normalize
        self.size = size
        self.apply_exposure = apply_exposure
        self.cache_processed = cache_processed
        self.cache_dir = os.path.join(test_dir, "..", ".cache", f"test_s{size}_exp{int(apply_exposure)}")
        self.image_files = sorted(
            [f for f in os.listdir(test_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        )

    def _cache_path(self, image_path: str) -> str:
        rel = os.path.relpath(image_path, self.test_dir)
        stem, _ = os.path.splitext(rel)
        return os.path.join(self.cache_dir, stem + ".npy")

    def _get_image(self, image_path: str) -> np.ndarray:
        if not self.cache_processed:
            return _load_and_resize(image_path, self.size, self.apply_exposure)

        cache_path = self._cache_path(image_path)
        if os.path.exists(cache_path):
            return np.load(cache_path)

        image = _load_and_resize(image_path, self.size, self.apply_exposure)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, image)
        return image

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index: int):
        file_name = self.image_files[index]
        image_path = os.path.join(self.test_dir, file_name)
        image = self._get_image(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        if self.normalize:
            image = image / 255.0
        return image, file_name


def get_dataloaders(
    data_root: str = ".",
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.2,
    image_size: int = 32,
    normalize: bool = True,
    apply_exposure: bool = True,
    cache_processed: bool = True,
    seed: int = 42,
    load_train: bool = True,
    load_val: bool = True,
    load_test: bool = True,
):
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = None
    val_loader = None
    test_loader = None

    if load_train or load_val:
        train_dir = _resolve_train_dir(data_root)
        dataset = ImageDataset(
            root_dir=train_dir,
            normalize=normalize,
            size=image_size,
            apply_exposure=apply_exposure,
            cache_processed=cache_processed,
        )
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        if load_train:
            train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        if load_val:
            val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    if load_test:
        test_dir = _resolve_test_dir(data_root)
        test_set = TestImageDataset(
            test_dir=test_dir,
            normalize=normalize,
            size=image_size,
            apply_exposure=apply_exposure,
            cache_processed=cache_processed,
        )
        test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
