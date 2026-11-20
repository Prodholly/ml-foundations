import numpy as np
import pytest

from numpy_mlp.data import label_one_hot
from numpy_mlp.losses import softmax_cross_entropy
from numpy_mlp.model import NumpyMLP
from numpy_mlp.regularization import Dropout, add_penalty_grads, l1_grad, l1_penalty, penalty


@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
def test_dropout_mask_fraction_of_zeros(p):
    drop = Dropout(p, rng=np.random.default_rng(0))
    Y = drop.forward(np.ones((500, 400)), training=True)
    assert abs(np.mean(Y == 0) - p) < 0.01


def test_dropout_keeps_the_expected_value():
    drop = Dropout(0.5, rng=np.random.default_rng(0))
    assert abs(drop.forward(np.ones((500, 400))).mean() - 1.0) < 0.01


def test_dropout_eval_is_identity_and_backward_uses_the_mask():
    drop = Dropout(0.5, rng=np.random.default_rng(0))
    X = np.arange(12.0).reshape(3, 4)
    np.testing.assert_array_equal(drop.forward(X, training=False), X)

    drop.forward(np.ones((3, 4)), training=True)
    grad = drop.backward(np.ones((3, 4)))
    np.testing.assert_array_equal(grad, drop.mask)


def test_l2_grad_matches_numerical_with_regularization(rng, grad_check):
    lam = 0.05
    model = NumpyMLP([6, 5, 4, 3], activation="swish", init_scheme="he", seed=1)
    X, Y = rng.normal(size=(7, 6)), label_one_hot(rng.integers(0, 3, 7), 3)

    def total_loss():
        data_loss, _ = softmax_cross_entropy(model.forward(X), Y)
        return data_loss + penalty(model, l2_lambda=lam)

    model.zero_grad()
    _, dZ = softmax_cross_entropy(model.forward(X), Y)
    model.backward(dZ)
    add_penalty_grads(model, l2_lambda=lam)

    for layer in model.layers:
        W0 = layer.W.copy()

        def f(W, layer=layer):
            layer.W = W
            return total_loss()

        err = grad_check(f, W0, layer.dW.copy())
        layer.W = W0
        assert err < 1e-5


def test_l1_penalty_and_grad():
    W = np.array([[1.0, -2.0], [0.0, 3.0]])
    assert l1_penalty([W], 0.1) == pytest.approx(0.6)
    np.testing.assert_array_equal(l1_grad(W, 0.1), [[0.1, -0.1], [0.0, 0.1]])


def test_dropout_rejects_p_of_one():
    with pytest.raises(ValueError):
        Dropout(1.0)
