import numpy as np
import pytest

from numpy_mlp.layers import LinearLayer

N, IN, OUT = 4, 5, 3


@pytest.fixture
def setup(rng):
    layer = LinearLayer(IN, OUT, init_scheme="he", rng=rng)
    layer.b = rng.normal(size=OUT)
    X = rng.normal(size=(N, IN))
    upstream = rng.normal(size=(N, OUT))
    return layer, X, upstream


def _loss(layer, X, upstream):
    return float(np.sum(upstream * layer.forward(X)))


def test_grad_W(setup, grad_check):
    layer, X, up = setup
    layer.forward(X)
    layer.backward(up)

    def f(W):
        layer.W = W
        return _loss(layer, X, up)

    W0 = layer.W.copy()
    err = grad_check(f, W0, layer.dW.copy())
    layer.W = W0
    assert err < 1e-5


def test_grad_b(setup, grad_check):
    layer, X, up = setup
    layer.forward(X)
    layer.backward(up)

    def f(b):
        layer.b = b
        return _loss(layer, X, up)

    assert grad_check(f, layer.b.copy(), layer.db.copy()) < 1e-5


def test_grad_X(setup, grad_check):
    layer, X, up = setup
    layer.forward(X)
    dX = layer.backward(up)
    assert grad_check(lambda X_: _loss(layer, X_, up), X, dX) < 1e-5


def test_shapes(setup):
    layer, X, up = setup
    Z = layer.forward(X)
    dX = layer.backward(up)
    assert Z.shape == (N, OUT)
    assert dX.shape == X.shape == (N, IN)
    assert layer.dW.shape == layer.W.shape == (OUT, IN)
    assert layer.db.shape == layer.b.shape == (OUT,)


def test_grads_accumulate_until_zero_grad(setup):
    layer, X, up = setup
    layer.forward(X)
    layer.backward(up)
    once = layer.dW.copy()
    layer.backward(up)
    np.testing.assert_allclose(layer.dW, 2 * once)
    layer.zero_grad()
    assert not layer.dW.any() and not layer.db.any()


def test_wrong_input_width_raises(setup):
    layer, _, _ = setup
    with pytest.raises(ValueError):
        layer.forward(np.ones((2, IN + 1)))
