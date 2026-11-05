import numpy as np
import pytest

from numpy_ops import (
    add_bias,
    inspect_tensor,
    loop_elementwise_mult,
    loop_row_mean,
    numpy_elementwise_mult,
    numpy_row_mean,
    outer_add,
    reduce_and_annotate,
)


# >> task 1: shape inspection

def test_inspect_tensor_prints_every_field(capsys):
    inspect_tensor(np.ones((2, 3, 4)), "batch")  # shape: (2, 3, 4)
    out = capsys.readouterr().out
    assert "batch" in out
    assert "(2, 3, 4)" in out
    assert "ndim: 3" in out
    assert "size: 24" in out
    assert "float64" in out


def test_scalar_shape_is_empty_tuple():
    s = np.array(5)  # shape: ()
    assert s.shape == ()  # not (1,)
    assert s.ndim == 0
    assert s.size == 1


def test_inspect_tensor_scalar(capsys):
    inspect_tensor(np.array(5), "scalar")  # shape: ()
    out = capsys.readouterr().out
    assert "shape: ()" in out
    assert "ndim: 0" in out


def test_inspect_tensor_rejects_list():
    # a plain list has no .shape
    with pytest.raises(AttributeError):
        inspect_tensor([1, 2, 3], "list")


# >> task 2: reshape & indexing

def test_reshape_preserves_elements():
    x = np.arange(12)    # shape: (12,)
    X = x.reshape(3, 4)  # shape: (3, 4)
    assert X.shape == (3, 4)
    assert X.size == x.size  # 12 elements either way


def test_reshape_is_row_major():
    x = np.arange(12)        # shape: (12,)
    X = x.reshape(3, 4)      # shape: (3, 4)
    Y = x.reshape(2, 2, 3)   # shape: (2, 2, 3)
    assert x[11] == X[2, 3] == Y[1, 1, 2]
    assert x[4] == X[1, 0]   # second row starts at element 4


def test_reshape_infers_minus_one():
    X = np.arange(12).reshape(-1, 4)  # shape: (3, 4)
    assert X.shape == (3, 4)


def test_reshape_wrong_size_raises():
    # 12 elements can't fill a (2, 3) grid
    with pytest.raises(ValueError):
        np.arange(12).reshape(2, 3)


# >> task 3: axis operations

def test_sum_axis0_collapses_rows():
    X = np.ones((2, 3))     # shape: (2, 3)
    result = X.sum(axis=0)  # shape: (3,)
    assert result.shape == (3,)


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_reductions_shapes(op):
    X = np.arange(6).reshape(2, 3)  # shape: (2, 3)
    f = getattr(X, op)
    assert f().shape == ()          # all axes collapsed
    assert f(axis=0).shape == (3,)
    assert f(axis=1).shape == (2,)


def test_keepdims_keeps_axis_as_one():
    X = np.ones((2, 3))                         # shape: (2, 3)
    assert X.sum(axis=1, keepdims=True).shape == (2, 1)


def test_reduce_and_annotate_prints_shapes(capsys):
    reduce_and_annotate(np.ones((2, 3)))  # shape: (2, 3)
    out = capsys.readouterr().out
    for shape in ["(2, 3)", "()", "(3,)", "(2,)", "(2, 1)"]:
        assert shape in out


def test_axis_out_of_range_raises():
    # a 2-D array has no axis 2
    with pytest.raises(np.exceptions.AxisError):
        np.ones((2, 3)).sum(axis=2)


# >> task 4: broadcasting

def test_broadcast_bias_addition():
    A = np.ones((3, 4))      # shape: (3, 4)
    b = np.ones((4,))        # shape: (4,)
    result = add_bias(A, b)  # shape: (3, 4)
    assert result.shape == (3, 4)


def test_bias_is_added_to_every_row():
    A = np.zeros((3, 4))           # shape: (3, 4)
    b = np.arange(4)               # shape: (4,)
    result = add_bias(A, b)        # shape: (3, 4)
    assert np.array_equal(result, np.tile(b, (3, 1)))


def test_outer_add_stretches_both():
    r = np.arange(4).reshape(1, 4)       # shape: (1, 4)
    c = np.arange(3).reshape(3, 1)       # shape: (3, 1)
    result = outer_add(r, c)             # shape: (3, 4)
    assert result.shape == (3, 4)
    assert result[2, 3] == 2 + 3


def test_incompatible_shapes_raise():
    # (3, 4) + (3,) should raise because trailing dims don't align
    with pytest.raises(ValueError):
        np.ones((3, 4)) + np.ones((3,))


# >> task 5: loop vs vectorized

def test_elementwise_mult_loop_matches_numpy():
    rng = np.random.default_rng(0)
    A = rng.random((5, 7))  # shape: (5, 7)
    B = rng.random((5, 7))  # shape: (5, 7)
    loop_result = loop_elementwise_mult(A, B)    # shape: (5, 7)
    numpy_result = numpy_elementwise_mult(A, B)  # shape: (5, 7)
    assert loop_result.shape == numpy_result.shape == (5, 7)
    assert np.allclose(loop_result, numpy_result)


def test_row_mean_loop_matches_numpy():
    A = np.random.default_rng(0).random((5, 7))  # shape: (5, 7)
    loop_means = loop_row_mean(A)    # shape: (5,)
    numpy_means = numpy_row_mean(A)  # shape: (5,)
    assert loop_means.shape == numpy_means.shape == (5,)
    assert np.allclose(loop_means, numpy_means)


def test_single_element_matrix():
    A = np.array([[3.0]])  # shape: (1, 1)
    assert loop_elementwise_mult(A, A).shape == (1, 1)
    assert loop_row_mean(A).shape == (1,)
    assert loop_row_mean(A)[0] == 3.0


def test_mismatched_shapes_raise():
    A = np.ones((3, 3))  # shape: (3, 3)
    B = np.ones((2, 2))  # shape: (2, 2)
    with pytest.raises(ValueError):
        numpy_elementwise_mult(A, B)
    # the loop version walks past the end of B
    with pytest.raises(IndexError):
        loop_elementwise_mult(A, B)
