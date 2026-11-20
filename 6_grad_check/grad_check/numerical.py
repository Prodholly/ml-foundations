import numpy as np


def numerical_gradient(f, x: np.ndarray, h: float = 1e-05) -> np.ndarray:
    x = x.astype(float)
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"])

    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]

        x[ix] = old_value + h
        f_add = f(x)

        x[ix] = old_value - h
        f_sub = f(x)

        x[ix] = old_value
        grad[ix] = (f_add - f_sub) / (2 * h)

        it.iternext()

    return grad


if __name__ == "__main__":
    some_feature = np.arange(1, 5)
    centered_diff = numerical_gradient(lambda x: np.sum(x**2), some_feature)
    print("centered difference for each value:", centered_diff)
