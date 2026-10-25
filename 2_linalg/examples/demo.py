"""Demo: this walks through matmul, softmax and cross-entropy on a tiny classifier."""

import time

import numpy as np
from linalg.ops import matmul
from linalg.einsum_matmul import matmul_einsum
from linalg.einsum_softmax import softmax_einsum
from linalg.cross_entropy import cross_entropy

# seeded so every run prints the same numbers
rng = np.random.default_rng(42)

print(">>> 1. Matmul: loop vs einsum vs NumPy <<<")
A = rng.random((3, 2))
B = rng.random((2, 4))
C = matmul(A, B)
print("A is", A.shape, "B is", B.shape, "so A @ B is", C.shape)
print("loop matches numpy?  ", np.allclose(C, A @ B))
print("einsum matches numpy?", np.allclose(matmul_einsum(A, B), A @ B))


print(">>> 2. A tiny classifier: 3 samples, 4 features, 3 classes <<<")
X = rng.random((3, 4))  # one row per sample
w = rng.random((3, 4))  # one weight vector per class
z = matmul(X, w.T)      # logits, shape (3, 3): one score per class per sample
print("Logits:\n", z)


print(">>> 3. Softmax turns each row of logits into probabilities <<<")
probs = softmax_einsum(z)
print("Probabilities:\n", probs)
print("Each row sums to:", probs.sum(axis=1))  # should be [1, 1, 1]


print(">>> 4. Cross-entropy: how wrong are those probabilities? <<<")
y_true = np.eye(3)  # sample 0 is class 0, sample 1 is class 1, sample 2 is class 2
uniform = np.full((3, 3), 1 / 3)  # a model that just guesses evenly
print(f"Loss with random weights:   {cross_entropy(y_true, probs):.4f}")
print(f"Loss from guessing evenly:  {cross_entropy(y_true, uniform):.4f}")  # ln(3) = 1.0986
print(f"Loss from a 90% confident correct model: "
      f"{cross_entropy(y_true, np.where(y_true == 1, 0.9, 0.05)):.4f}")


print(">>> 5. Why we don't use the loop in practice <<<")
# 200x200 keeps this demo quick; linalg/benchmark.py uses 500x500
A = rng.random((200, 200))
B = rng.random((200, 200))
for name, fn in [("loop", matmul), ("einsum", matmul_einsum), ("numpy", np.matmul)]:
    start = time.perf_counter()
    fn(A, B)
    print(f"{name:>6}: {time.perf_counter() - start:.4f} seconds")
