import numpy as np
import pytest
from linalg.ops import matmul
from linalg.einsum_matmul import matmul_einsum
from linalg.einsum_softmax import softmax_einsum
from linalg.softmax import softmax


# --- matmul tests ---

def test_matmul_matches_numpy():
    A = np.random.rand(3, 2)
    B = np.random.rand(2, 4)
    assert np.allclose(matmul(A, B), np.matmul(A, B))

def test_matmul_output_shape():
    A = np.random.rand(3, 2)
    B = np.random.rand(2, 4)
    assert matmul(A, B).shape == (3, 4)

def test_matmul_einsum_matches_numpy():
    A = np.random.rand(3, 2)
    B = np.random.rand(2, 4)
    assert np.allclose(matmul_einsum(A, B), np.matmul(A, B))


# --- softmax tests ---

def test_softmax_sums_to_one():
    z = np.random.rand(3, 3)
    probs = softmax_einsum(z)
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_softmax_all_positive():
    z = np.random.rand(3, 3)
    probs = softmax_einsum(z)
    assert np.all(probs > 0)

def test_softmax_highest_logit_gets_highest_prob():
    z = np.array([[10.0, 1.0, 1.0]])
    probs = softmax_einsum(z)
    assert probs[0, 0] == probs[0].max()