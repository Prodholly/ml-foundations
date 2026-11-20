from __future__ import annotations

import numpy as np

from .activations import softmax

EPS = 1e-9


def _check_pair(A: np.ndarray, B: np.ndarray, a: str, b: str) -> None:
    if A.shape != B.shape:
        raise ValueError(f"{a} and {b} must have the same shape, got {A.shape} and {B.shape}")
    if A.ndim != 2:
        raise ValueError(f"{a} must be 2-d (N, C), got shape {A.shape}")


def cross_entropy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    _check_pair(Y_pred, Y_true, "Y_pred", "Y_true")
    N = Y_pred.shape[0]
    return float(-np.sum(Y_true * np.log(Y_pred + EPS)) / N)


def softmax_cross_entropy_grad(Z: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    _check_pair(Z, Y_true, "Z", "Y_true")
    N = Z.shape[0]
    return (softmax(Z) - Y_true) / N


def softmax_cross_entropy(Z: np.ndarray, Y_true: np.ndarray) -> tuple[float, np.ndarray]:
    return cross_entropy(softmax(Z), Y_true), softmax_cross_entropy_grad(Z, Y_true)


def mse(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    _check_pair(Y_pred, Y_true, "Y_pred", "Y_true")
    return float(np.mean((Y_pred - Y_true) ** 2))


def mse_grad(Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    _check_pair(Y_pred, Y_true, "Y_pred", "Y_true")
    return 2.0 * (Y_pred - Y_true) / Y_pred.size
