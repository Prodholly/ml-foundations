import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grad_check.numerical import numerical_gradient
from grad_check.analytical import softmax, cross_entropy_loss, forward, analytical_gradient_W


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)

    z_normal = rng.standard_normal((10, 5))
    p_normal = softmax(z_normal)
    np.testing.assert_allclose(
        p_normal.sum(axis=1), np.ones(10), atol=1e-6,
        err_msg="softmax rows should sum to 1 (normal inputs)"
    )

    z_large = rng.standard_normal((10, 5)) * 1000
    p_large = softmax(z_large)
    np.testing.assert_allclose(
        p_large.sum(axis=1), np.ones(10), atol=1e-6,
        err_msg="softmax rows should sum to 1 (large inputs, stability)"
    )


def test_cross_entropy_perfect_prediction():
    N, C = 4, 3
    labels = np.array([0, 1, 2, 0])
    y_true = np.eye(C)[labels]

    probs = y_true * (1 - 1e-9) + 1e-9 / C
    probs = probs / probs.sum(axis=1, keepdims=True)

    loss = cross_entropy_loss(probs, y_true)

    assert loss < 1e-5, f"loss should be near 0 for perfect predictions, got {loss:.6f}"


def test_cross_entropy_uniform_prediction():
    N, C = 6, 4
    labels = np.arange(N) % C
    y_true = np.eye(C)[labels]
    probs = np.full((N, C), 1.0 / C)

    loss = cross_entropy_loss(probs, y_true)
    expected = np.log(C)

    np.testing.assert_allclose(
        loss, expected, rtol=1e-4,
        err_msg=f"uniform predictions should give loss = log(C) = {expected:.4f}"
    )


def test_analytical_grad_shape_matches_W():
    N, D, C = 7, 5, 4
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, D))
    W = rng.standard_normal((D, C))
    labels = rng.integers(0, C, size=N)
    y_true = np.eye(C)[labels]

    probs, _ = forward(X, W)
    grad = analytical_gradient_W(X, probs, y_true)

    assert grad.shape == W.shape, (
        f"gradient shape {grad.shape} does not match W shape {W.shape}"
    )


def test_gradient_check_passes():
    np.random.seed(42)
    N, D, C = 5, 4, 3

    X = np.random.randn(N, D)
    W = np.random.randn(D, C)
    labels = np.array([0, 1, 2, 0, 1])
    y_true = np.eye(C)[labels]

    def loss_fn(W_flat):
        probs, _ = forward(X, W_flat.reshape(D, C))
        return cross_entropy_loss(probs, y_true)

    num_grad = numerical_gradient(loss_fn, W.flatten()).reshape(D, C)
    probs, _ = forward(X, W)
    ana_grad = analytical_gradient_W(X, probs, y_true)

    numerator = np.abs(ana_grad - num_grad)
    denom = np.maximum(np.abs(ana_grad), np.abs(num_grad)) + 1e-8
    rel_error = numerator / denom

    assert rel_error.max() < 1e-5, (
        f"gradient check failed: max relative error = {rel_error.max():.2e}\n"
        f"worst element:\n"
        f"  analytical = {ana_grad.flat[rel_error.argmax()]:.8f}\n"
        f"  numerical  = {num_grad.flat[rel_error.argmax()]:.8f}"
    )


def test_numerical_gradient_int_input():
    x = np.arange(1, 5)
    grad = numerical_gradient(lambda v: np.sum(v**2), x)
    np.testing.assert_allclose(grad, 2 * x, rtol=1e-6)


def test_numerical_gradient_scalar():
    x = np.array(3.0)
    grad = numerical_gradient(lambda v: v**2, x)
    assert grad.shape == ()
    np.testing.assert_allclose(grad, 6.0, rtol=1e-6)


def test_numerical_gradient_leaves_input_alone():
    x = np.array([1.0, 2.0, 3.0])
    before = x.copy()
    numerical_gradient(lambda v: np.sum(v**3), x)
    np.testing.assert_array_equal(x, before)
