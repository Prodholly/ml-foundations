import numpy as np


# Softmax: convert logits to probabilities
def softmax(z):
    # subtract max for numerical stability which prevents exp() from overflowing
    # result is mathematically identical, but safe for large values
    shifted = z - np.max(z)

    # exponentiate every element
    exp_z = np.exp(shifted)

    # divide each by the total so all probabilities sum to 1
    prob = exp_z / np.sum(exp_z)
    return prob


if __name__ == "__main__":
    # lets simulate a small dataset and weights
    # X: 3 samples, each with 4 features
    X = np.random.rand(3, 4)

    # w: 1 weight vector with 4 values (one per feature)
    w = np.random.rand(1, 4)

    # Forward pass: compute raw scores (logits)
    # w.T transposes w from (1,4) to (4,1) so dot product works
    # z shape: (3,1) — one score per sample
    z = np.dot(X, w.T)
    print("Logits:\n", z)

    probs = softmax(z)
    print("Probabilities:\n", probs)

    # just to check: probabilities must sum to 1.0
    print("Sum of probs:", np.sum(probs))
