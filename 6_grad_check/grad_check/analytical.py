import numpy as np


# 1. softmax
def softmax(z: np.ndarray) -> np.ndarray:
    """
    z: (N, C), N samples and C classes
    returns: (N, C), each row sums to 1
    """
    z_stable = z - np.max(z, axis=1, keepdims=True)  # (N, C) subtract the row max so exp() can't overflow
    exp_z = np.exp(z_stable)
    return exp_z / exp_z.sum(axis=1, keepdims=True)    # (N, C)


# 2. cross-entropy
def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    probs:  (N, C), output from softmax
    y_true: (N, C), one-hot labels
    returns: scalar, mean loss over the batch
    """
    N = probs.shape[0]
    # i could have written it in two steps too:
    #   log_probs = np.log(probs + 1e-9)        # (N, C), +1e-9 so log(0) never happens
    #   return -np.sum(y_true * log_probs) / N
    return -np.sum(y_true * np.log(probs + 1e-9)) / N


# 3. forward pass
def forward(X: np.ndarray, W: np.ndarray) -> tuple:
    """
    X: (N, D)   W: (D, C)
    returns: probs (N, C), z (N, C)
    """
    z = X @ W   # (N, D) @ (D, C) = (N, C); throws if W's shape doesn't line up
    probs = softmax(z)  # (N, C)
    return probs, z


# 4. analytical gradient
def analytical_gradient_W(X: np.ndarray, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    gradient of cross_entropy(softmax(X @ W)) w.r.t. W.
    softmax + cross-entropy together simplify to:
        dL/dz = (probs - y_true) / N
        dL/dW = X.T @ dL/dz

    X:      (N, D)
    probs:  (N, C)
    y_true: (N, C), one-hot
    returns: (D, C), same shape as W
    """
    N = X.shape[0]
    delta = (probs - y_true) / N  # (N, C)
    return X.T @ delta # (D, N) @ (N, C) = (D, C)
