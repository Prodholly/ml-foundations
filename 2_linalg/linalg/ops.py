import numpy as np



# problem
# Matrix multiply: Given two matrices A and B, compute their product


def matmul(X, Y):
    # the inner dimensions must match: (m, n) @ (n, p); without this check a mismatch silently gives wrong numbers
    if X.shape[1] != Y.shape[0]:
        raise ValueError(f"inner dimensions don't match: {X.shape} @ {Y.shape}")

    rows_X = X.shape[0] # this determines the rows of the output C
    cols_Y = Y.shape[1] # this determines the cols of the output C
    inner = Y.shape[0] # this is the inner dimensions which determines how many matching pairs are added
    C = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(rows_X): # produces 0, 1, 2
        for j in range(cols_Y): # produces 0, 1, 2
            for k in range(inner):
                C[i, j] += X[i, k] * Y[k, j]

    return C


if __name__ == "__main__":
    A = np.random.rand(3, 2)
    B = np.random.rand(2, 3)

    results = matmul(A, B)
    print(results)
    print("they match?", np.allclose(results, np.matmul(A, B)))
