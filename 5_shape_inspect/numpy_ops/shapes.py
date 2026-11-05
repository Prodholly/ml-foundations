import time

import numpy as np


# <<< 1 shape inspection

def inspect_tensor(x: np.ndarray, name: str) -> None:
    print(f"{name} | shape: {x.shape} | ndim: {x.ndim} | size: {x.size} | dtype: {x.dtype}")


def shape_demo() -> None:
    # create several tensors of different shapes
    tensor0 = np.array(5)               # shape: ()        -- 0D scalar
    tensor1 = np.array([1, 2, 3, 4, 5]) # shape: (5,)      -- 1D
    tensor2 = np.ones((3, 4))           # shape: (3, 4)    -- 2D
    tensor3 = np.ones((2, 3, 4))        # shape: (2, 3, 4) -- 3D

    inspect_tensor(tensor0, "scalar")
    inspect_tensor(tensor1, "vector")
    inspect_tensor(tensor2, "matrix")
    inspect_tensor(tensor3, "batch")


# <<< 2 reshape & indexing

def reshape_demo() -> None:
    # create a 1-D array
    x = np.arange(12)  # shape: (12,)
    print("Original array:", x)
    print("Original shape:", x.shape)

    # reshape into a 3 by 4 matrix
    X = x.reshape(3, 4)       # shape: (3, 4)
    print("\nReshaped into (3, 4):")
    print(X)
    print("Shape:", X.shape)

    # reshape into a 3-D array
    Y = x.reshape(2, 2, 3)    # shape: (2, 2, 3)
    print("\nReshaped into (2, 2, 3):")
    print(Y)
    print("Shape:", Y.shape)

    # access the same element using different indexing methods
    a = x[11]   # shape: ()
    b = X[2, 3]         # shape: ()
    c = Y[1, 1, 2]  # shape: ()

    print("\nelement comparison:")
    print("x[11]:", a)
    print("X[2, 3]:", b)
    print("Y[1, 1, 2]:", c)

    # assert that all three elements are equal
    assert a == b == c


# <<< 3 axis operations

def reduce_and_annotate(X: np.ndarray) -> None:
    # X: (2, 3)
    print(f"input shape: {X.shape}")
    print(f"sum() shape: {X.sum().shape}")  # shape: (), scalar
    print(f"sum(axis=0) shape: {X.sum(axis=0).shape}")        # shape: (3,), rows collapse, keep columns
    print(f"sum(axis=1) shape: {X.sum(axis=1).shape}")    # shape: (2,), columns collapse, keep rows
    print(f"sum(axis=1, keepdims=True) shape: {X.sum(axis=1, keepdims=True).shape}")  # shape: (2, 1), reduced axis kept as size 1


def axis_demo() -> None:
    u = np.arange(6).reshape(2, 3)  # shape: (2, 3)

    reduce_and_annotate(u)

    # same reductions for mean and max
    for op in ("mean", "max"):
        f = getattr(u, op)
        print(f"\n{op}() shape: {f().shape}")        # shape: ()
        print(f"{op}(axis=0) shape: {f(axis=0).shape}")    # shape: (3,)
        print(f"{op}(axis=1) shape: {f(axis=1).shape}")    # shape: (2,)


# <<< 4 broadcasting

def add_bias(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A: (3, 4), b: (4,) -> b broadcast across rows
    return A + b                    # shape: (3, 4)


def outer_add(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    # r: (1, 4), c: (3, 1) -> both stretch
    return r + c                    # shape: (3, 4)


def broadcast_demo() -> None:
    A = np.arange(12).reshape(3, 4)          # shape: (3, 4)
    b = np.ones((4,))                        # shape: (4,)
    r = np.arange(4).reshape(1, 4)           # shape: (1, 4)
    column_vect = np.array([[5], [6], [7]])  # shape: (3, 1)

    # prediction: (3, 4) + (4,) -> (3, 4)
    print(f"Bias addition: A {A.shape} + b {b.shape}")
    result1 = add_bias(A, b)                 # shape: (3, 4)
    print(result1)
    print("Shape:", result1.shape)

    # prediction: (1, 4) + (3, 1) -> (3, 4)
    print(f"\nOuter addition: r {r.shape} + c {column_vect.shape}")
    result2 = outer_add(r, column_vect)      # shape: (3, 4)
    print(result2)
    print("Shape:", result2.shape)


# <<< 5 loop vs vectorized benchmark

def loop_elementwise_mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n, m = A.shape
    result = np.zeros((n, m))  # shape: (n, m)

    for i in range(n):
        for j in range(m):
            result[i, j] = A[i, j] * B[i, j]

    return result  # shape: (n, m)


def numpy_elementwise_mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A * B  # shape: same as inputs


def loop_row_mean(A: np.ndarray) -> np.ndarray:
    n, m = A.shape
    result = np.zeros(n)  # shape: (n,)

    for i in range(n):
        total = 0.0
        for j in range(m):
            total += A[i, j]
        result[i] = total / m

    return result  # shape: (n,)


def numpy_row_mean(A: np.ndarray) -> np.ndarray:
    return A.mean(axis=1)  # shape: (n,)


def benchmark_demo() -> None:
    # create two large arrays
    rng = np.random.default_rng()
    A = rng.random((1000, 1000))  # shape: (1000, 1000)
    B = rng.random((1000, 1000))  # shape: (1000, 1000)

    # element-wise multiplication
    start = time.perf_counter()
    loop_result = loop_elementwise_mult(A, B) # shape: (1000, 1000)
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    numpy_result = numpy_elementwise_mult(A, B)  # shape: (1000, 1000)
    numpy_time = time.perf_counter() - start

    assert np.allclose(loop_result, numpy_result)

    print("ELEMENT-WISE MULTIPLICATION")
    print(f"Python loops: {loop_time:.6f} seconds")
    print(f"NumPy: {numpy_time:.6f} seconds")
    print(f"Speedup:{loop_time / numpy_time:.2f}x")

    # row-wise mean
    start = time.perf_counter()
    loop_means = loop_row_mean(A) # shape: (1000,)
    loop_mean_time = time.perf_counter() - start

    start = time.perf_counter()
    numpy_means = numpy_row_mean(A) # shape: (1000,)
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
