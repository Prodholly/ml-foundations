# run from 6_grad_check:  python3 examples/run_all.py
import os
import sys
import timeit

import matplotlib
matplotlib.use("Agg")   # no window, just save the file
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grad_check import numerical_gradient
from grad_check import softmax, cross_entropy_loss, forward, analytical_gradient_W

FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")


# 1. numerical gradient on a 1d vector
def check_vector():
    x_vec = np.arange(1, 5, dtype=float)  # (4,)
    grad_vec = numerical_gradient(lambda x: np.sum(x**2), x_vec)
    print("input vector:", x_vec)
    print("numerical gradient:", grad_vec)
    print("expected (2x):", 2 * x_vec)


# 2. numerical gradient on a 2d matrix
def check_matrix():
    W = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])     # (2, 3) some weight
    grad_W = numerical_gradient(lambda w: np.sum(w**2), W)
    print("input matrix shape:", W.shape)
    print("numerical gradient:\n", grad_W)
    print("expected (2W):\n", 2 * W)


# 3. forward pass
def setup():
    np.random.seed(42)
    N, D, C = 5, 4, 3   # 5 samples, 4 features, 3 classes
    X = np.random.randn(N, D) # (5, 4)
    W = np.random.randn(D, C)  # (4, 3)

    # one-hot labels: each of the 5 samples belongs to one of 3 classes
    labels = np.array([0, 1, 2, 0, 1])
    y_true = np.eye(C)[labels]    # (5, 3)
    return X, W, y_true


def check_forward(X, W, y_true):
    probs, z = forward(X, W)
    loss = cross_entropy_loss(probs, y_true)

    print("shapes:")
    print(f"  X: {X.shape}")
    print(f"  W: {W.shape}")
    print(f"  z: {z.shape}   (logits = X @ W)")
    print(f"  probs: {probs.shape}  (after softmax)")
    print(f"  y_true: {y_true.shape} (one-hot)")
    print(f"\nloss: {loss:.4f}")
    print(f"\nprobs (each row sums to 1):\n{probs.round(3)}")
    print(f"row sums: {probs.sum(axis=1).round(6)}")


# 4. gradient check: numerical vs analytical dL/dW
def check_gradient(X, W, y_true, threshold=1e-5):
    D, C = W.shape

    # X and y_true stay fixed, W is the thing we differentiate
    def loss_fn(W_flat: np.ndarray) -> float:
        probs, _ = forward(X, W_flat.reshape(D, C))  # (5, 3)
        return cross_entropy_loss(probs, y_true)     # scalar

    # numerical: slow but trustworthy
    num_grad = numerical_gradient(loss_fn, W.flatten())  # (12,)
    num_grad = num_grad.reshape(D, C)	# (4, 3)

    # analytical: fast, the one we're checking
    probs, _ = forward(X, W)   # (5, 3)
    ana_grad = analytical_gradient_W(X, probs, y_true)  # (4, 3)

    # relative error, element by element
    numerator = np.abs(ana_grad - num_grad)  # (4, 3)
    denominator = np.maximum(np.abs(ana_grad), np.abs(num_grad)) + 1e-8
    rel_error = numerator / denominator    # (4, 3)

    print(f"analytical gradient:\n{ana_grad.round(6)}")
    print(f"\nnumerical gradient:\n{num_grad.round(6)}")
    print(f"\nrelative error (element-wise):\n{rel_error}")
    print(f"\nmax relative error:  {rel_error.max():.2e}")
    print(f"mean relative error: {rel_error.mean():.2e}")

    if rel_error.max() < threshold:
        print(f"\npass: max error {rel_error.max():.2e} < {threshold:.0e}")
    else:
        print(f"\nfail: max error {rel_error.max():.2e} >= {threshold:.0e}")


# 5. loop vs numpy benchmark
def benchmark(sizes=(100, 1_000, 10_000, 100_000, 1_000_000)):
    loop_times = []
    numpy_times = []

    print(f"{'n':>12}  {'loop (s)':>12}  {'numpy (s)':>12}  {'speedup':>10}")
    print("-" * 52)

    for n in sizes:
        A = np.random.rand(n)
        B = np.random.rand(n)

        # python loop, one run per timing
        def loop_mult():
            result = np.zeros(n)
            for i in range(n):
                result[i] = A[i] * B[i]
            return result

        # numpy, 10 runs per timing because one run is too fast to measure
        def numpy_mult():
            return A * B

        # min of 3 repeats, the min is the most stable number (noise only ever adds time)
        loop_t = min(timeit.repeat(loop_mult, number=1, repeat=3))
        numpy_t = min(timeit.repeat(numpy_mult, number=10, repeat=3)) / 10

        loop_times.append(loop_t)
        numpy_times.append(numpy_t)
        print(f"{n:>12,}  {loop_t:>12.4f}  {numpy_t:>12.6f}  {loop_t / numpy_t:>9.0f}x")

    return list(sizes), loop_times, numpy_times


def plot_speedup(sizes, loop_times, numpy_times):
    speedups = [lt / nt for lt, nt in zip(loop_times, numpy_times)]

    os.makedirs(FIGURES, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    # left: the two raw times, slow one red, fast one green
    left.loglog(sizes, loop_times, marker="o", linewidth=2, color="#d62828", label="python loop (slow)")
    left.loglog(sizes, numpy_times, marker="o", linewidth=2, color="#2a9d3f", label="numpy (fast)")
    left.fill_between(sizes, numpy_times, loop_times, color="gray", alpha=0.08)  # the gap = the speedup
    left.set_xlabel("array size (log scale)")
    left.set_ylabel("time per run, seconds (log scale)")
    left.set_title("loop vs numpy: time")
    left.legend()
    left.grid(True, which="both", alpha=0.3)

    # right: the ratio between them
    right.loglog(sizes, speedups, marker="o", linewidth=2, color="steelblue")
    for x, y in zip(sizes, speedups):     # label each point with its speedup
        right.annotate(f"{y:.0f}x", xy=(x, y), xytext=(5, 5),
                       textcoords="offset points", fontsize=8)
    right.set_xlabel("array size (log scale)")
    right.set_ylabel("speedup  loop time / numpy time  (log scale)")
    right.set_title("numpy speedup vs array size")
    right.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES, "speedup_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("\nfigure saved -> figures/speedup_curve.png")


if __name__ == "__main__":
    print("=== 1. 1d vector ===")
    check_vector()

    print("\n=== 2. 2d matrix ===")
    check_matrix()

    print("\n=== 3. forward pass ===")
    X, W, y_true = setup()
    check_forward(X, W, y_true)

    print("\n=== 4. gradient check: dL/dW ===")
    check_gradient(X, W, y_true)

    print("\n=== 5. vectorization benchmark ===")
    plot_speedup(*benchmark())
