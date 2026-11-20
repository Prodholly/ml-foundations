from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from .model import NumpyMLP


def l2_penalty(weights: Iterable[np.ndarray], lam: float) -> float:
    return float(lam * sum(np.sum(W**2) for W in weights))


def l2_grad(W: np.ndarray, lam: float) -> np.ndarray:
    return 2.0 * lam * W


def l1_penalty(weights: Iterable[np.ndarray], lam: float) -> float:
    return float(lam * sum(np.sum(np.abs(W)) for W in weights))


def l1_grad(W: np.ndarray, lam: float) -> np.ndarray:
    return lam * np.sign(W)


def penalty(model: "NumpyMLP", l2_lambda: float = 0.0, l1_lambda: float = 0.0) -> float:
    weights = [layer.W for layer in model.layers]
    total = 0.0
    if l2_lambda:
        total += l2_penalty(weights, l2_lambda)
    if l1_lambda:
        total += l1_penalty(weights, l1_lambda)
    return total


def add_penalty_grads(model: "NumpyMLP", l2_lambda: float = 0.0, l1_lambda: float = 0.0) -> None:
    for layer in model.layers:
        if l2_lambda:
            layer.dW += l2_grad(layer.W, l2_lambda)
        if l1_lambda:
            layer.dW += l1_grad(layer.W, l1_lambda)


class Dropout:
    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p
        self.rng = np.random.default_rng() if rng is None else rng
        self.training = True
        self.mask: np.ndarray | None = None

    def forward(self, X: np.ndarray, training: bool | None = None) -> np.ndarray:
        training = self.training if training is None else training

        if not training or self.p == 0.0:
            self.mask = None
            return X

        keep = 1.0 - self.p
        bernoulli = self.rng.random(X.shape) < keep
        self.mask = bernoulli / keep
        return X * self.mask

    def backward(self, dL_dY: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return dL_dY
        return dL_dY * self.mask

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
