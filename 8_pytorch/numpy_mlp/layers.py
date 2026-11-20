from __future__ import annotations

import numpy as np

from .init_schemes import init_weights


class LinearLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_scheme: str = "he",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.init_scheme = init_scheme

        self.W = init_weights(init_scheme, fan_in=in_features, fan_out=out_features, rng=rng)
        self.b = np.zeros(out_features)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._X: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.in_features:
            raise ValueError(f"expected X of shape (N, {self.in_features}), got {X.shape}")
        self._X = X
        return X @ self.W.T + self.b

    def backward(self, dL_dZ: np.ndarray) -> np.ndarray:
        if self._X is None:
            raise RuntimeError("backward called before forward")
        if dL_dZ.shape != (self._X.shape[0], self.out_features):
            raise ValueError(
                f"expected dL_dZ of shape ({self._X.shape[0]}, {self.out_features}), got {dL_dZ.shape}"
            )

        self.dW += dL_dZ.T @ self._X
        self.db += dL_dZ.sum(axis=0)
        return dL_dZ @ self.W

    def zero_grad(self) -> None:
        self.dW.fill(0.0)
        self.db.fill(0.0)

    @property
    def num_params(self) -> int:
        return self.W.size + self.b.size

    def __repr__(self) -> str:
        return f"LinearLayer({self.in_features} -> {self.out_features}, init={self.init_scheme})"
