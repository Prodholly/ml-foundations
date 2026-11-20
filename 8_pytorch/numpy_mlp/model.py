from __future__ import annotations

from typing import Iterator

import numpy as np

from .activations import Activation, get_activation, softmax
from .layers import LinearLayer
from .regularization import Dropout


class NumpyMLP:
    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        init_scheme: str = "he",
        dropout_p: float = 0.0,
        seed: int = 0,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError(f"need at least [in, out], got {layer_sizes}")

        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.init_scheme = init_scheme
        self.dropout_p = dropout_p
        self.training = True
        rng = np.random.default_rng(seed)

        self.layers: list[LinearLayer] = [
            LinearLayer(fan_in, fan_out, init_scheme=init_scheme, rng=rng)
            for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.activations: list[Activation] = [get_activation(activation) for _ in self.layers[:-1]]
        self.dropouts: list[Dropout] = (
            [Dropout(dropout_p, rng=rng) for _ in self.layers[:-1]] if dropout_p > 0 else []
        )


    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        for i, layer in enumerate(self.layers):
            Z = layer.forward(A)
            if i == len(self.layers) - 1:
                return Z
            A = self.activations[i].forward(Z)
            if self.dropouts:
                A = self.dropouts[i].forward(A, training=self.training)
        return A

    def backward(self, dL_dZ_last: np.ndarray) -> np.ndarray:
        grad = self.layers[-1].backward(dL_dZ_last)
        for i in reversed(range(len(self.layers) - 1)):
            if self.dropouts:
                grad = self.dropouts[i].backward(grad)
            grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)
        return grad


    def parameters(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        for layer in self.layers:
            yield layer.W, layer.b, layer.dW, layer.db

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def num_params(self) -> int:
        return sum(layer.num_params for layer in self.layers)

    def get_state(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def set_state(self, state: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for layer, (W, b) in zip(self.layers, state):
            layer.W[...] = W
            layer.b[...] = b


    def train(self) -> "NumpyMLP":
        self.training = True
        return self

    def eval(self) -> "NumpyMLP":
        self.training = False
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            return softmax(self.forward(X))
        finally:
            self.training = was_training

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def __repr__(self) -> str:
        arch = " -> ".join(map(str, self.layer_sizes))
        return (
            f"NumpyMLP({arch}, activation={self.activation_name}, init={self.init_scheme}, "
            f"dropout={self.dropout_p}, params={self.num_params():,})"
        )
