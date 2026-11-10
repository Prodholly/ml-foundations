# tests/test_grad_check.py
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grad_check.numerical import numerical_gradient
from grad_check.analytical import softmax, cross_entropy_loss, forward, analytical_gradient_W


# 1. softmax rows sum to 1
def test_softmax_rows_sum_to_one():
    """
    softmax output is a probability distribution, so every row must sum to 1.
    checks normal inputs and huge ones (numerical stability).
    """
    rng = np.random.default_rng(0)

    # normal inputs
    z_normal = rng.standard_normal((10, 5))  # (10, 5)
    p_normal = softmax(z_normal)
    np.testing.assert_allclose(
        p_normal.sum(axis=1), np.ones(10), atol=1e-6,
        err_msg="softmax rows should sum to 1 (normal inputs)"
    )

    # large inputs, without the max-subtraction trick exp() overflows to inf
    z_large = rng.standard_normal((10, 5)) * 1000
    p_large = softmax(z_large)
    np.testing.assert_allclose(
        p_large.sum(axis=1), np.ones(10), atol=1e-6,
        err_msg="softmax rows should sum to 1 (large inputs, stability)"
    )


# 2. perfect prediction -> loss near 0
def test_cross_entropy_perfect_prediction():
    N, C = 4, 3
    labels = np.array([0, 1, 2, 0])
    y_true = np.eye(C)[labels]   # (4, 3)

    # almost exactly one-hot, still a valid distribution
    probs = y_true * (1 - 1e-9) + 1e-9 / C
    probs = probs / probs.sum(axis=1, keepdims=True)

    loss = cross_entropy_loss(probs, y_true)

    assert loss < 1e-5, f"loss should be near 0 for perfect predictions, got {loss:.6f}"


# 3. uniform prediction -> loss = log(C)
def test_cross_entropy_uniform_prediction():
    """
    uniform predictions give loss = log(C).
    that's the "knows nothing" baseline: every class equally likely.
    """
    N, C = 6, 4
    labels = np.arange(N) % C   # [0, 1, 2, 3, 0, 1]
    y_true = np.eye(C)[labels]  # (6, 4)
    probs = np.full((N, C), 1.0 / C)    # (6, 4) each class 0.25

    loss = cross_entropy_loss(probs, y_true)
    expected = np.log(C)  # log(4) ≈ 1.386

    np.testing.assert_allclose(
        loss, expected, rtol=1e-4,
        err_msg=f"uniform predictions should give loss = log(C) = {expected:.4f}"
    )


# 4. gradient has the same shape as W
def test_analytical_grad_shape_matches_W():
    """
    dL/dW must be the same shape as W, a mismatch means a broadcasting slip.
    """
    N, D, C = 7, 5, 4
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, D))   # (7, 5)
    W = rng.standard_normal((D, C)) # (5, 4)
    labels = rng.integers(0, C, size=N)
    y_true = np.eye(C)[labels]  # (7, 4)

    probs, _ = forward(X, W)
    grad = analytical_gradient_W(X, probs, y_true)

    assert grad.shape == W.shape, (
        f"gradient shape {grad.shape} does not match W shape {W.shape}"
    )


# 5. the actual gradient check
def test_gradient_check_passes():
    """
    analytical gradient must match the numerical one within relative error 1e-5.
    if this fails, the bug is in analytical.py or numerical.py.
    """
    np.random.seed(42)
    N, D, C = 5, 4, 3

    X = np.random.randn(N, D)  # (5, 4)
    W = np.random.randn(D, C)  # (4, 3)
    labels = np.array([0, 1, 2, 0, 1])
    y_true = np.eye(C)[labels]	# (5, 3)

    # closure: X and y_true fixed, W is what we differentiate
    def loss_fn(W_flat):
        probs, _ = forward(X, W_flat.reshape(D, C))
        return cross_entropy_loss(probs, y_true)

    num_grad = numerical_gradient(loss_fn, W.flatten()).reshape(D, C)  # (4, 3)
    probs, _ = forward(X, W)
    ana_grad = analytical_gradient_W(X, probs, y_true)    # (4, 3)

    numerator = np.abs(ana_grad - num_grad)
    denom = np.maximum(np.abs(ana_grad), np.abs(num_grad)) + 1e-8
    rel_error = numerator / denom

    assert rel_error.max() < 1e-5, (
        f"gradient check failed: max relative error = {rel_error.max():.2e}\n"
        f"worst element:\n"
        f"  analytical = {ana_grad.flat[rel_error.argmax()]:.8f}\n"
        f"  numerical  = {num_grad.flat[rel_error.argmax()]:.8f}"
    )


# 6. integer input still works (ints can't hold x ± 1e-5)
def test_numerical_gradient_int_input():
    x = np.arange(1, 5)  # int64
    grad = numerical_gradient(lambda v: np.sum(v**2), x)
    np.testing.assert_allclose(grad, 2 * x, rtol=1e-6)


# 7. scalar (0-d) input
def test_numerical_gradient_scalar():
    x = np.array(3.0)   # shape ()
    grad = numerical_gradient(lambda v: v**2, x)
    assert grad.shape == ()
    np.testing.assert_allclose(grad, 6.0, rtol=1e-6)


# 8. the input array isn't changed
def test_numerical_gradient_leaves_input_alone():
    x = np.array([1.0, 2.0, 3.0])
    before = x.copy()
    numerical_gradient(lambda v: np.sum(v**3), x)
    np.testing.assert_array_equal(x, before)
