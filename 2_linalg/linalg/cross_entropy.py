import numpy as np
from linalg.softmax import softmax


def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))


if __name__ == "__main__":
    X = np.random.rand(3, 4)

    # Three classes, four features per sample
    w = np.random.rand(3, 4)

    # Forward pass: one logit per class per sample
    z = np.dot(X, w.T)  # shape: (3, 3)

    # Softmax across classes for each sample
    y_pred = softmax(z)

    # One-hot targets: one correct class per sample
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    loss = cross_entropy(y_true, y_pred)
    print("Cross entropy loss:", loss)
