"""train / validation / test splitting and a standardizer that can only be fitted on one split"""

from __future__ import annotations

import numpy as np

from prob_stats._validation import as_1d_float, as_2d_float, as_labels

__all__ = ["train_val_test_split", "Standardizer"]


def train_val_test_split(
    n_samples: int,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    shuffled, disjoint (train, val, test) index arrays covering range(n_samples)
    train fits weights, val picks hyperparameters, test is touched once at the end
    stratify keeps the positive rate equal across the three; it splits rows, so it's unsafe for grouped data
    """
    if n_samples < 3:
        raise ValueError(f"n_samples must be >= 3, got {n_samples}")
    for name, frac in (("val_frac", val_frac), ("test_frac", test_frac)):
        if not 0.0 < frac < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {frac}")
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac + test_frac must be < 1, got "
            f"{val_frac + test_frac}"
        )

    rng = np.random.default_rng(seed)

    def split_block(indices: np.ndarray) -> tuple[np.ndarray, ...]:
        """shuffle one block of indices and cut it three ways"""
        perm = rng.permutation(indices)  # shape: (m,)
        m = perm.shape[0]
        n_test = int(round(test_frac * m))
        n_val = int(round(val_frac * m))
        # guarantee a non-empty train set even for tiny blocks
        n_val = min(n_val, max(0, m - n_test - 1))
        return (
            perm[n_test + n_val :],  # train
            perm[n_test : n_test + n_val],  # val
            perm[:n_test],  # test
        )

    if stratify is None:
        train_idx, val_idx, test_idx = split_block(np.arange(n_samples))
    else:
        labels = as_labels(stratify, name="stratify")  # shape: (n,)
        if labels.shape[0] != n_samples:
            raise ValueError(
                f"stratify has length {labels.shape[0]}, expected "
                f"{n_samples}"
            )
        blocks = [
            split_block(np.flatnonzero(labels == c)) for c in (0.0, 1.0)
        ]
        train_idx = np.concatenate([b[0] for b in blocks])
        val_idx = np.concatenate([b[1] for b in blocks])
        test_idx = np.concatenate([b[2] for b in blocks])

    # re-shuffle so class order inside each split is not an artifact of
    # stratification, which would matter for any order-sensitive
    # downstream step
    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    test_idx = rng.permutation(test_idx)

    if train_idx.size == 0:
        raise ValueError(
            f"train split is empty with n_samples={n_samples}, "
            f"val_frac={val_frac}, test_frac={test_frac}"
        )

    return (
        train_idx.astype(np.int64),
        val_idx.astype(np.int64),
        test_idx.astype(np.int64),
    )


class Standardizer:
    """
    per-feature (x - mean) / std, fitted on training rows only
    no fit_transform on purpose: fitting on everything before splitting is the classic leak
    constant features get scale 1.0 instead of a divide by zero, and I count them in n_constant_
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None  # shape: (d,)
        self.scale_: np.ndarray | None = None  # shape: (d,)
        self.n_constant_: int | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        """learn per-feature mean and std (1/n) from X; pass training rows only"""
        X_arr = as_2d_float(X, name="X")  # shape: (n, d)

        self.mean_ = X_arr.mean(axis=0)  # shape: (d,)
        std = X_arr.std(axis=0, ddof=0)  # shape: (d,), MLE convention

        constant = std == 0.0  # shape: (d,), bool
        self.n_constant_ = int(np.count_nonzero(constant))

        self.scale_ = np.where(constant, 1.0, std)  # shape: (d,)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """apply the fitted centering and scaling; shape (n, d) -> (n, d)"""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "Standardizer is not fitted; call fit(X_train) first. "
                "There is no fit_transform on purpose -- fitting on the "
                "full dataset leaks evaluation data into the model."
            )
        X_arr = as_2d_float(X, name="X")  # shape: (n, d)

        if X_arr.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but this Standardizer "
                f"was fitted on {self.mean_.shape[0]}"
            )
        return (X_arr - self.mean_) / self.scale_  # shape: (n, d)
