import numpy as np
import pytest

from numpy_mlp.activations import (
    GELU, ReLU, Swish, gelu, gelu_backward, relu, relu_backward, sigmoid, softmax, swish, swish_backward,
)


def test_relu_values():
    assert relu(np.array(0.0)) == 0
    assert relu(np.array(-1.0)) == 0
    assert relu(np.array(1.0)) == 1


def _away_from_kink(rng, shape):
    x = rng.normal(size=shape)
    x[np.abs(x) < 1e-3] = 0.5
    return x


@pytest.mark.parametrize(
    "f, f_backward",
    [(relu, relu_backward), (gelu, gelu_backward), (swish, swish_backward)],
    ids=["relu", "gelu", "swish"],
)
def test_activation_gradient_check(f, f_backward, rng, grad_check):
    x = _away_from_kink(rng, (6, 5))
    upstream = rng.normal(size=x.shape)

    loss = lambda x_: float(np.sum(upstream * f(x_)))
    assert grad_check(loss, x, f_backward(x, upstream)) < 1e-5


def test_relu_gradient_is_dead_for_negative_inputs():
    x = np.array([-2.0, -0.1, 0.1, 2.0])
    np.testing.assert_array_equal(relu_backward(x, np.ones(4)), [0, 0, 1, 1])


def test_gelu_and_swish_gradients_do_not_die():
    x = np.array([-2.0, -1.0, -0.5])
    assert np.all(gelu_backward(x, np.ones(3)) != 0)
    assert np.all(swish_backward(x, np.ones(3)) != 0)


def test_gelu_close_to_exact_erf_form():
    from math import erf, sqrt

    x = np.linspace(-3, 3, 61)
    exact = np.array([v * 0.5 * (1 + erf(v / sqrt(2))) for v in x])
    assert np.max(np.abs(gelu(x) - exact)) < 1e-3


def test_sigmoid_is_stable_for_large_inputs():
    with np.errstate(over="raise"):
        s = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0])


def test_softmax_rows_sum_to_one_and_survive_big_logits():
    p = softmax(np.array([[1000.0, 1000.0, 0.0], [1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.all(np.isfinite(p))


@pytest.mark.parametrize("cls, f", [(ReLU, relu), (GELU, gelu), (Swish, swish)])
def test_classes_cache_input_and_match_functions(cls, f, rng):
    act, x = cls(), rng.normal(size=(3, 4))
    np.testing.assert_allclose(act.forward(x), f(x))
    assert act.backward(np.ones_like(x)).shape == x.shape


def test_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        ReLU().backward(np.ones(3))
