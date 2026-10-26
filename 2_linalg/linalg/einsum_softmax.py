import numpy as np


# Original softmax (for comparison)
from linalg.softmax import softmax


# einsum softmax
def softmax_einsum(z):
    # subtract each row's own max; a single global max can push a whole row to exp() = 0 and give 0/0 = nan
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))

    # sum across classes (axis=1) → one total per sample
    row_sums = exp_z.sum(axis=1)  # shape: (3,)

    # divide each row of exp_z by its corresponding row_sum
    # 'ij,i->ij' means: for each element [i,j], divide by row_sum[i]
    prob = np.einsum('ij,i->ij', exp_z, 1 / row_sums)
    return prob


if __name__ == "__main__":
    # 3 samples, 3 classes, 4 features — proper multiclass setup
    X = np.random.rand(3, 4)
    w = np.random.rand(3, 4)  # one weight vector per class

    # Forward pass: one logit per class per sample
    # z shape: (3, 3) — 3 samples, 3 class scores each
    z = np.dot(X, w.T)
    print("Logits:\n", z)

    probs = softmax_einsum(z)
    print("Probabilities:\n", probs)
    print("Sum per sample:", probs.sum(axis=1))  # should be [1, 1, 1]
