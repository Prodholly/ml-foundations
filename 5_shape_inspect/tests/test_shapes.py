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


def test_inspect_tensor_prints_every_field(capsys):
    inspect_tensor(np.ones((2, 3, 4)), "batch")
    out = capsys.readouterr().out
    assert "batch" in out
    assert "(2, 3, 4)" in out
    assert "ndim: 3" in out
    assert "size: 24" in out
    assert "float64" in out


def test_scalar_shape_is_empty_tuple():
    s = np.array(5)
    assert s.shape == ()
    assert s.ndim == 0
    assert s.size == 1


def test_inspect_tensor_scalar(capsys):
    inspect_tensor(np.array(5), "scalar")
    out = capsys.readouterr().out
    assert "shape: ()" in out
    assert "ndim: 0" in out


def test_inspect_tensor_rejects_list():
    with pytest.raises(AttributeError):
        inspect_tensor([1, 2, 3], "list")


def test_reshape_preserves_elements():
    x = np.arange(12)
    X = x.reshape(3, 4)
    assert X.shape == (3, 4)
    assert X.size == x.size


def test_reshape_is_row_major():
    x = np.arange(12)
    X = x.reshape(3, 4)
    Y = x.reshape(2, 2, 3)
    assert x[11] == X[2, 3] == Y[1, 1, 2]
    assert x[4] == X[1, 0]


def test_reshape_infers_minus_one():
    X = np.arange(12).reshape(-1, 4)
    assert X.shape == (3, 4)


def test_reshape_wrong_size_raises():
    with pytest.raises(ValueError):
        np.arange(12).reshape(2, 3)


def test_sum_axis0_collapses_rows():
    X = np.ones((2, 3))
    result = X.sum(axis=0)
    assert result.shape == (3,)


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_reductions_shapes(op):
    X = np.arange(6).reshape(2, 3)
    f = getattr(X, op)
    assert f().shape == ()
    assert f(axis=0).shape == (3,)
    assert f(axis=1).shape == (2,)


def test_keepdims_keeps_axis_as_one():
    X = np.ones((2, 3))
    assert X.sum(axis=1, keepdims=True).shape == (2, 1)


def test_reduce_and_annotate_prints_shapes(capsys):
    reduce_and_annotate(np.ones((2, 3)))
    out = capsys.readouterr().out
    for shape in ["(2, 3)", "()", "(3,)", "(2,)", "(2, 1)"]:
        assert shape in out


def test_axis_out_of_range_raises():
    with pytest.raises(np.exceptions.AxisError):
        np.ones((2, 3)).sum(axis=2)


def test_broadcast_bias_addition():
    A = np.ones((3, 4))
    b = np.ones((4,))
    result = add_bias(A, b)
    assert result.shape == (3, 4)


def test_bias_is_added_to_every_row():
    A = np.zeros((3, 4))
    b = np.arange(4)
    result = add_bias(A, b)
    assert np.array_equal(result, np.tile(b, (3, 1)))


def test_outer_add_stretches_both():
    r = np.arange(4).reshape(1, 4)
    c = np.arange(3).reshape(3, 1)
    result = outer_add(r, c)
    assert result.shape == (3, 4)
    assert result[2, 3] == 2 + 3


def test_incompatible_shapes_raise():
    with pytest.raises(ValueError):
        np.ones((3, 4)) + np.ones((3,))


def test_elementwise_mult_loop_matches_numpy():
    rng = np.random.default_rng(0)
    A = rng.random((5, 7))
    B = rng.random((5, 7))
    loop_result = loop_elementwise_mult(A, B)
    numpy_result = numpy_elementwise_mult(A, B)
    assert loop_result.shape == numpy_result.shape == (5, 7)
    assert np.allclose(loop_result, numpy_result)


def test_row_mean_loop_matches_numpy():
    A = np.random.default_rng(0).random((5, 7))
    loop_means = loop_row_mean(A)
    numpy_means = numpy_row_mean(A)
    assert loop_means.shape == numpy_means.shape == (5,)
    assert np.allclose(loop_means, numpy_means)


def test_single_element_matrix():
    A = np.array([[3.0]])
    assert loop_elementwise_mult(A, A).shape == (1, 1)
    assert loop_row_mean(A).shape == (1,)
    assert loop_row_mean(A)[0] == 3.0


def test_mismatched_shapes_raise():
    A = np.ones((3, 3))
    B = np.ones((2, 2))
    with pytest.raises(ValueError):
        numpy_elementwise_mult(A, B)
    with pytest.raises(IndexError):
        loop_elementwise_mult(A, B)
