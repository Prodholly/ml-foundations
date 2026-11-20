from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import NumpyMLP


def _flatten(model: "NumpyMLP") -> tuple[list[np.ndarray], list[np.ndarray]]:
    params, grads = [], []
    for W, b, dW, db in model.parameters():
        params += [W, b]
        grads += [dW, db]
    return params, grads


class Optimizer:
    def __init__(self, lr: float) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        self.lr = lr

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        raise NotImplementedError

    def step(self, model: "NumpyMLP") -> None:
        self.update(*_flatten(model))

    @staticmethod
    def _check(params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if len(params) != len(grads):
            raise ValueError(f"{len(params)} params but {len(grads)} grads")
        for p, g in zip(params, grads):
            if p.shape != g.shape:
                raise ValueError(f"param shape {p.shape} != grad shape {g.shape}")


class SGDMomentum(Optimizer):
    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(lr)
        self.beta = beta
        self.velocity: list[np.ndarray] | None = None

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self._check(params, grads)
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]

        for p, g, v in zip(params, grads, self.velocity):
            v *= self.beta
            v -= self.lr * g
            p += v


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0
        self.m: list[np.ndarray] | None = None
        self.v: list[np.ndarray] | None = None

    def _adam_step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t

        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g**2
            m_hat = m / bias1
            v_hat = v / bias2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self._check(params, grads)
        self._adam_step(params, grads)


class AdamW(Adam):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self._check(params, grads)
        for p in params:
            p *= 1.0 - self.lr * self.weight_decay
        self._adam_step(params, grads)
