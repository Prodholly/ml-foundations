import numpy as np
import pytest

from numpy_mlp.init_schemes import init_weights

FAN_IN, FAN_OUT = 784, 256


def test_he_variance(rng):
    W = init_weights("he", FAN_IN, FAN_OUT, rng)
    target = 2.0 / FAN_IN
    assert abs(W.var() - target) / target < 0.10


def test_xavier_variance(rng):
    W = init_weights("xavier", FAN_IN, FAN_OUT, rng)
    target = 2.0 / (FAN_IN + FAN_OUT)
    assert abs(W.var() - target) / target < 0.10


def test_lecun_variance(rng):
    W = init_weights("lecun", FAN_IN, FAN_OUT, rng)
    target = 1.0 / FAN_IN
    assert abs(W.var() - target) / target < 0.10


def test_gaussian_baseline_std(rng):
    assert abs(init_weights("gaussian", FAN_IN, FAN_OUT, rng).std() - 0.01) < 0.001


def test_shape_is_out_by_in(rng):
    assert init_weights("he", 5, 3, rng).shape == (3, 5)


def test_unknown_scheme_raises(rng):
    with pytest.raises(ValueError):
        init_weights("kaiming_magic", 4, 4, rng)
