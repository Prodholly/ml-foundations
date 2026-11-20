import numpy as np

from numpy_mlp.model import NumpyMLP
from numpy_mlp.optimizers import Adam, AdamW, SGDMomentum


def test_sgd_step_moves_against_the_gradient():
    theta, grad = np.array([1.0, -2.0]), np.array([0.5, -0.5])
    SGDMomentum(lr=0.1, beta=0.9).update([theta], [grad])
    np.testing.assert_allclose(theta, [0.95, -1.95])


def test_sgd_momentum_builds_velocity():
    theta, grad = np.array([0.0]), np.array([1.0])
    opt = SGDMomentum(lr=0.1, beta=0.9)
    opt.update([theta], [grad])
    opt.update([theta], [grad])
    np.testing.assert_allclose(theta, [-0.29])


def test_adam_bias_correction_at_t1():
    theta, grad = np.array([0.0, 0.0]), np.array([4.0, -0.003])
    opt = Adam(lr=0.01)
    opt.update([theta], [grad])
    assert opt.t == 1
    np.testing.assert_allclose(theta, [-0.01, 0.01], rtol=1e-4)


def test_adamw_decay_is_independent_of_gradient():
    theta, zero_grad = np.array([2.0, -4.0]), np.zeros(2)
    AdamW(lr=0.1, weight_decay=0.5).update([theta], [zero_grad])
    np.testing.assert_allclose(theta, [1.9, -3.8])


def test_adam_with_zero_gradient_does_not_move():
    theta = np.array([2.0, -4.0])
    Adam(lr=0.1).update([theta], [np.zeros(2)])
    np.testing.assert_allclose(theta, [2.0, -4.0])


def test_step_updates_model_weights_in_place(rng):
    model = NumpyMLP([4, 3, 2], seed=0)
    for layer in model.layers:
        layer.dW[...] = 1.0
    W_before = model.layers[0].W.copy()
    SGDMomentum(lr=0.1).step(model)
    np.testing.assert_allclose(model.layers[0].W, W_before - 0.1)


def test_optimizers_minimize_a_quadratic():
    for opt in (SGDMomentum(lr=0.05), Adam(lr=0.1), AdamW(lr=0.1, weight_decay=0.0)):
        theta = np.zeros(3)
        for _ in range(500):
            opt.update([theta], [2 * (theta - 3.0)])
        np.testing.assert_allclose(theta, 3.0, atol=1e-2)
