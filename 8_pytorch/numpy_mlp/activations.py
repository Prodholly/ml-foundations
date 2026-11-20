from __future__ import annotations

import numpy as np

_GELU_C = np.sqrt(2.0 / np.pi)
_GELU_A = 0.044715


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_backward(x: np.ndarray, dL_dy: np.ndarray) -> np.ndarray:
    return dL_dy * (x > 0)


def gelu(x: np.ndarray) -> np.ndarray:
    u = _GELU_C * (x + _GELU_A * x**3)
    return 0.5 * x * (1.0 + np.tanh(u))


def gelu_backward(x: np.ndarray, dL_dy: np.ndarray) -> np.ndarray:
    u = _GELU_C * (x + _GELU_A * x**3)
    t = np.tanh(u)
    du_dx = _GELU_C * (1.0 + 3.0 * _GELU_A * x**2)
    dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du_dx
    return dL_dy * dy_dx


def swish(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def swish_backward(x: np.ndarray, dL_dy: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return dL_dy * (x * s * (1.0 - s) + s)


def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class Activation:
    name = "base"

    def __init__(self) -> None:
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return self._f(x)

    def backward(self, dL_dy: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError(f"{self.name}.backward called before forward")
        return self._df(self._x, dL_dy)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def _f(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _df(self, x: np.ndarray, dL_dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    name = "relu"

    def _f(self, x):
        return relu(x)

    def _df(self, x, dL_dy):
        return relu_backward(x, dL_dy)


class GELU(Activation):
    name = "gelu"

    def _f(self, x):
        return gelu(x)

    def _df(self, x, dL_dy):
        return gelu_backward(x, dL_dy)


class Swish(Activation):
    name = "swish"

    def _f(self, x):
        return swish(x)

    def _df(self, x, dL_dy):
        return swish_backward(x, dL_dy)


ACTIVATIONS: dict[str, type[Activation]] = {"relu": ReLU, "gelu": GELU, "swish": Swish}


def get_activation(name: str) -> Activation:
    try:
        return ACTIVATIONS[name.lower()]()
    except KeyError:
        raise ValueError(f"unknown activation {name!r}, choose from {sorted(ACTIVATIONS)}") from None
