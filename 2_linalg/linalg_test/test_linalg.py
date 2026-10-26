import numpy as np
import pytest
from linalg.ops import matmul
from linalg.einsum_matmul import matmul_einsum
from linalg.einsum_softmax import softmax_einsum
from linalg.softmax import softmax
from linalg.cross_entropy import cross_entropy

# seeded so a failing test can be reproduced
rng = np.random.default_rng(0)


# --- matmul tests ---

def test_matmul_matches_numpy():
    A = rng.random((3, 2))
    B = rng.random((2, 4))
    assert np.allclose(matmul(A, B), np.matmul(A, B))

def test_matmul_output_shape():
    A = rng.random((3, 2))
    B = rng.random((2, 4))
    assert matmul(A, B).shape == (3, 4)

def test_matmul_einsum_matches_numpy():
    A = rng.random((3, 2))
    B = rng.random((2, 4))
    assert np.allclose(matmul_einsum(A, B), np.matmul(A, B))

def test_matmul_rejects_mismatched_shapes():
    A = rng.random((3, 4))
    B = rng.random((2, 3))
    with pytest.raises(ValueError):
        matmul(A, B)


# --- softmax tests ---

def test_softmax_sums_to_one():
    z = rng.random((3, 3))
    probs = softmax_einsum(z)
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_softmax_all_positive():
    z = rng.random((3, 3))
    probs = softmax_einsum(z)
    assert np.all(probs > 0)

def test_softmax_highest_logit_gets_highest_prob():
    z = np.array([[10.0, 1.0, 1.0]])
    probs = softmax_einsum(z)
    assert probs[0, 0] == probs[0].max()

def test_softmax_einsum_stable_with_rows_on_different_scales():
    z = np.array([[1000.0, 0.0], [0.0, -1000.0]])
    probs = softmax_einsum(z)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_softmax_normalizes_each_row():
    z = rng.random((3, 3))
    probs = softmax(z)
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_softmax_matches_softmax_einsum():
    z = rng.random((3, 3))
    assert np.allclose(softmax(z), softmax_einsum(z))


# --- cross-entropy tests ---

def test_cross_entropy_uniform_guess_is_log_num_classes():
    y_true = np.eye(3)
    uniform = np.full((3, 3), 1 / 3)
    assert np.isclose(cross_entropy(y_true, uniform), np.log(3))
