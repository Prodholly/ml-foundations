import numpy as np
from numpy import einsum


# original: manual three-loop matmul
def matmul(X, Y):
    rows_X = X.shape[0]
    cols_Y = Y.shape[1]
    inner  = Y.shape[0]
    C = np.zeros((rows_X, cols_Y))
    for i in range(rows_X):
        for j in range(cols_Y):
            for k in range(inner):
                C[i, j] += X[i, k] * Y[k, j]
    return C


# einsum version which has same operation but in just one line
def matmul_einsum(X, Y):
    return einsum('ik,kj->ij', X, Y)


if __name__ == "__main__":
    A = np.random.rand(3, 2)
    B = np.random.rand(2, 3)

    # Compare both
    result_manual = matmul(A, B)
    result_einsum = matmul_einsum(A, B)

    print("manual result:\n", result_manual)
    print("einsum result:\n", result_einsum)
    print("both match numpy?", np.allclose(result_manual, np.matmul(A, B)))
    print("einsum matches manual?", np.allclose(result_einsum, result_manual))
