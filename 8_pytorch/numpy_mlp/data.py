from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NUM_CLASSES = 10
NUM_PIXELS = 28 * 28

_IDX_LABELS = 2049
_IDX_IMAGES = 2051


def download_mnist(root: Path = DATA_DIR) -> Path:
    from torchvision.datasets import MNIST

    MNIST(root=str(root), train=True, download=True)
    MNIST(root=str(root), train=False, download=True)
    return Path(root) / "MNIST" / "raw"


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        raw = fh.read()

    magic = int.from_bytes(raw[0:4], "big")
    if magic == _IDX_LABELS:
        n = int.from_bytes(raw[4:8], "big")
        return np.frombuffer(raw, dtype=np.uint8, offset=8).reshape(n)
    if magic == _IDX_IMAGES:
        n, rows, cols = (int.from_bytes(raw[i:i + 4], "big") for i in (4, 8, 12))
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    raise ValueError(f"{path.name} is not an idx file (magic number {magic})")


def _find(raw_dir: Path, stem: str) -> Path:
    for candidate in (raw_dir / stem, raw_dir / f"{stem}.gz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{stem} not found in {raw_dir}")


def load_mnist(root: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_dir = Path(root) / "MNIST" / "raw"
    if not raw_dir.exists():
        raw_dir = download_mnist(root)

    images_train = _read_idx(_find(raw_dir, "train-images-idx3-ubyte"))
    labels_train = _read_idx(_find(raw_dir, "train-labels-idx1-ubyte"))
    images_test = _read_idx(_find(raw_dir, "t10k-images-idx3-ubyte"))
    labels_test = _read_idx(_find(raw_dir, "t10k-labels-idx1-ubyte"))

    X_train = images_train.reshape(-1, NUM_PIXELS).astype(np.float32)
    X_test = images_test.reshape(-1, NUM_PIXELS).astype(np.float32)

    return X_train, labels_train.astype(np.int64), X_test, labels_test.astype(np.int64)


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    n = X.shape[0]
    order = np.random.default_rng(seed).permutation(n)

    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    idx_train = order[:n_train]
    idx_val = order[n_train:n_train + n_val]
    idx_test = order[n_train + n_val:]

    return (X[idx_train], y[idx_train]), (X[idx_val], y[idx_val]), (X[idx_test], y[idx_test])


def fit_normalizer(X_train: np.ndarray) -> tuple[float, float]:
    mean = float(X_train.mean(dtype=np.float64))
    std = float(X_train.std(dtype=np.float64))
    if std == 0.0:
        raise ValueError("train split has zero variance, nothing to normalize")
    return mean, std


def normalize(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def label_one_hot(y_true: np.ndarray, num_labels: int = NUM_CLASSES) -> np.ndarray:
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-d (N,), got shape {y_true.shape}")
    if y_true.size and (y_true.min() < 0 or y_true.max() >= num_labels):
        raise ValueError(f"labels must be in [0, {num_labels}), got [{y_true.min()}, {y_true.max()}]")

    arr = np.zeros((y_true.shape[0], num_labels))
    arr[np.arange(y_true.shape[0]), y_true] = 1.0
    return arr


def batch_generator(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    n = X.shape[0]
    rng = np.random.default_rng(seed)

    while True:
        order = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            yield X[idx], Y[idx]


@dataclass
class MNISTSplits:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    X_final: np.ndarray
    Y_final: np.ndarray
    mean: float
    std: float


def prepare_mnist(root: Path = DATA_DIR, seed: int = 0) -> MNISTSplits:
    X_all, y_all, X_final, y_final = load_mnist(root)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_train_val_test(X_all, y_all, seed=seed)

    mean, std = fit_normalizer(X_tr)

    return MNISTSplits(
        X_train=normalize(X_tr, mean, std), Y_train=label_one_hot(y_tr),
        X_val=normalize(X_val, mean, std), Y_val=label_one_hot(y_val),
        X_test=normalize(X_te, mean, std), Y_test=label_one_hot(y_te),
        X_final=normalize(X_final, mean, std), Y_final=label_one_hot(y_final),
        mean=mean, std=std,
    )
