import numpy as np


# Softmax: convert logits to probabilities
# works on one sample's scores (1D) or a batch with one row per sample (2D); always normalizes the last axis
def softmax(z):
    # subtract max for numerical stability which prevents exp() from overflowing
    # result is mathematically identical, but safe for large values
    # axis=-1 takes each row's own max; keepdims=True keeps it shaped (rows, 1) so it lines up row by row
    shifted = z - np.max(z, axis=-1, keepdims=True)

    # exponentiate every element
    exp_z = np.exp(shifted)

    # divide each by its row's total so every sample's probabilities sum to 1
    prob = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    return prob


if __name__ == "__main__":
    # lets simulate one sample and a small set of class weights
    # x: 1 sample with 4 features
    x = np.random.rand(4)

    # w: 3 weight vectors with 4 values each (one per class)
    w = np.random.rand(3, 4)

    # Forward pass: compute raw scores (logits)
    # w @ x gives shape (3,) — one score per class for this sample
    z = w @ x
    print("Logits:\n", z)

    probs = softmax(z)
    print("Probabilities:\n", probs)

    # just to check: probabilities must sum to 1.0
    print("Sum of probs:", np.sum(probs))
