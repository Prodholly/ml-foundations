import time

import numpy as np


def inspect_tensor(x: np.ndarray, name: str) -> None:
    print(f"{name} | shape: {x.shape} | ndim: {x.ndim} | size: {x.size} | dtype: {x.dtype}")


def shape_demo() -> None:
    tensor0 = np.array(5)
    tensor1 = np.array([1, 2, 3, 4, 5])
    tensor2 = np.ones((3, 4))
    tensor3 = np.ones((2, 3, 4))

    inspect_tensor(tensor0, "scalar")
    inspect_tensor(tensor1, "vector")
    inspect_tensor(tensor2, "matrix")
    inspect_tensor(tensor3, "batch")


def reshape_demo() -> None:
    x = np.arange(12)
    print("Original array:", x)
    print("Original shape:", x.shape)

    X = x.reshape(3, 4)
    print("\nReshaped into (3, 4):")
    print(X)
    print("Shape:", X.shape)

    Y = x.reshape(2, 2, 3)
    print("\nReshaped into (2, 2, 3):")
    print(Y)
    print("Shape:", Y.shape)

    a = x[11]
    b = X[2, 3]
    c = Y[1, 1, 2]

    print("\nelement comparison:")
    print("x[11]:", a)
    print("X[2, 3]:", b)
    print("Y[1, 1, 2]:", c)

    assert a == b == c


def reduce_and_annotate(X: np.ndarray) -> None:
    print(f"input shape: {X.shape}")
    print(f"sum() shape: {X.sum().shape}")
    print(f"sum(axis=0) shape: {X.sum(axis=0).shape}")
    print(f"sum(axis=1) shape: {X.sum(axis=1).shape}")
    print(f"sum(axis=1, keepdims=True) shape: {X.sum(axis=1, keepdims=True).shape}")


def axis_demo() -> None:
    u = np.arange(6).reshape(2, 3)

    reduce_and_annotate(u)

    for op in ("mean", "max"):
        f = getattr(u, op)
        print(f"\n{op}() shape: {f().shape}")
        print(f"{op}(axis=0) shape: {f(axis=0).shape}")
        print(f"{op}(axis=1) shape: {f(axis=1).shape}")


def add_bias(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return A + b


def outer_add(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    return r + c


def broadcast_demo() -> None:
    A = np.arange(12).reshape(3, 4)
    b = np.ones((4,))
    r = np.arange(4).reshape(1, 4)
    column_vect = np.array([[5], [6], [7]])

    print(f"Bias addition: A {A.shape} + b {b.shape}")
    result1 = add_bias(A, b)
    print(result1)
    print("Shape:", result1.shape)

    print(f"\nOuter addition: r {r.shape} + c {column_vect.shape}")
    result2 = outer_add(r, column_vect)
    print(result2)
    print("Shape:", result2.shape)


def loop_elementwise_mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n, m = A.shape
    result = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            result[i, j] = A[i, j] * B[i, j]

    return result


def numpy_elementwise_mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A * B


def loop_row_mean(A: np.ndarray) -> np.ndarray:
    n, m = A.shape
    result = np.zeros(n)

    for i in range(n):
        total = 0.0
        for j in range(m):
            total += A[i, j]
        result[i] = total / m

    return result


def numpy_row_mean(A: np.ndarray) -> np.ndarray:
    return A.mean(axis=1)


def benchmark_demo() -> None:
    rng = np.random.default_rng()
    A = rng.random((1000, 1000))
    B = rng.random((1000, 1000))

    start = time.perf_counter()
    loop_result = loop_elementwise_mult(A, B)
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    numpy_result = numpy_elementwise_mult(A, B)
    numpy_time = time.perf_counter() - start

    assert np.allclose(loop_result, numpy_result)

    print("ELEMENT-WISE MULTIPLICATION")
    print(f"Python loops: {loop_time:.6f} seconds")
    print(f"NumPy: {numpy_time:.6f} seconds")
    print(f"Speedup:{loop_time / numpy_time:.2f}x")

    start = time.perf_counter()
    loop_means = loop_row_mean(A)
    loop_mean_time = time.perf_counter() - start

    start = time.perf_counter()
    numpy_means = numpy_row_mean(A)
    numpy_mean_time = time.perf_counter() - start

    assert np.allclose(loop_means, numpy_means)

    print("\nROW-WISE MEAN")
    print(f"Python loops: {loop_mean_time:.6f} seconds")
    print(f"NumPy: {numpy_mean_time:.6f} seconds")
    print(f"Speedup: {loop_mean_time / numpy_mean_time:.2f}x")


if __name__ == "__main__":
    for title, demo in [
        ("shape inspection", shape_demo),
        ("Reshape & Indexing", reshape_demo),
        ("axis Operations", axis_demo),
        ("Broadcasting", broadcast_demo),
        ("loop vs vectorized", benchmark_demo),
    ]:
        print(f"\n=== {title} ===")
        demo()
