import numpy as np


def numerical_gradient(f, x: np.ndarray, h: float = 1e-05) -> np.ndarray:
    """
    centered difference for every element of x:  (f(x + h) - f(x - h)) / 2h

    f: takes the whole array, returns a scalar
    x: any shape (1d, 2d, 3d ...)
    returns: same shape as x
    """
    x = x.astype(float)  # copy + float, ints can't hold x ± 1e-5 (they get truncated)
    grad = np.zeros_like(x)

    # np.nditer goes through each element regardless of the shape
    # multi_index gives (i,) or (i, j) or (i, j, k) at each step
    it = np.nditer(x, flags=["multi_index"])

    while not it.finished:
        ix = it.multi_index    # index of the current element
        old_value = x[ix]   # keep the original so we can put it back

        # 1. nudge up by h
        x[ix] = old_value + h
        f_add = f(x)  # scalar; the un-nudged values cancel out in the subtraction

        # 2. nudge down by h
        x[ix] = old_value - h
        f_sub = f(x)

        # 3. restore, then the slope
        x[ix] = old_value
        grad[ix] = (f_add - f_sub) / (2 * h)

        it.iternext()

    return grad


# quick sanity check: d/dx sum(x^2) = 2x
if __name__ == "__main__":
    some_feature = np.arange(1, 5)
    centered_diff = numerical_gradient(lambda x: np.sum(x**2), some_feature)
    print("centered difference for each value:", centered_diff)    # [2. 4. 6. 8.]
