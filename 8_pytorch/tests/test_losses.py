import numpy as np

from numpy_mlp.activations import softmax
from numpy_mlp.data import label_one_hot
from numpy_mlp.losses import cross_entropy, mse, mse_grad, softmax_cross_entropy_grad


def test_perfect_predictions_near_zero():
    Y = label_one_hot(np.array([0, 4, 9]))
    assert cross_entropy(Y.copy(), Y) < 1e-8


def test_uniform_predictions_give_log_10():
    Y = label_one_hot(np.array([0, 4, 9]))
    uniform = np.full((3, 10), 0.1)
    assert abs(cross_entropy(uniform, Y) - np.log(10)) < 1e-6


def test_softmax_ce_gradient_check(rng, grad_check):
    Z = rng.normal(size=(5, 10))
    Y = label_one_hot(rng.integers(0, 10, 5))
    loss = lambda Z_: cross_entropy(softmax(Z_), Y)
    assert grad_check(loss, Z, softmax_cross_entropy_grad(Z, Y)) < 1e-5


def test_softmax_ce_gradient_rows_sum_to_zero(rng):
    Z, Y = rng.normal(size=(4, 10)), label_one_hot(np.array([1, 2, 3, 4]))
    np.testing.assert_allclose(softmax_cross_entropy_grad(Z, Y).sum(axis=1), 0, atol=1e-12)


def test_mse_and_its_gradient(rng, grad_check):
    P, T = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    assert np.isclose(mse(P, P), 0.0)
    assert grad_check(lambda P_: mse(P_, T), P, mse_grad(P, T)) < 1e-7
