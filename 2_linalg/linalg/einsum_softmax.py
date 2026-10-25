import numpy as np


# Original softmax (for comparison)
def softmax(z):
    shifted = z - np.max(z)
    exp_z = np.exp(shifted)
    prob = exp_z / np.sum(exp_z)
    return prob


# einsum softmax
def softmax_einsum(z):
    exp_z = np.exp(z - np.max(z))

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
