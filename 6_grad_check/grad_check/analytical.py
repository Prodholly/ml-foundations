import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    N = probs.shape[0]
    return -np.sum(y_true * np.log(probs + 1e-9)) / N


def forward(X: np.ndarray, W: np.ndarray) -> tuple:
    z = X @ W
    probs = softmax(z)
    return probs, z


def analytical_gradient_W(X: np.ndarray, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    N = X.shape[0]
    delta = (probs - y_true) / N
    return X.T @ delta
