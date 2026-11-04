import pytest
import numpy as np

from utils import batch_items, check_shape, safe_parse_int

# >> tests for batch_items()

def test_batch_items_normal_split():
    assert batch_items([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_batch_items_empty_list():
    assert batch_items([], 3) == []


# match= checks it's my error: range() raises its own ValueError for a step of 0
@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_items_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        batch_items([1, 2, 3], batch_size)


def test_batch_items_batch_size_larger_than_list():
    assert batch_items([1, 2, 3], 10) == [[1, 2, 3]]


# >> tests for check_shape()

def test_check_shape_matching():
    tensor = np.zeros((3, 4))
    assert check_shape(tensor, (3, 4)) is True


@pytest.mark.parametrize("shape_dims", [(5, 4), (4, 2), (2, 4, 1)])
def test_check_shape_wrong(shape_dims):
    with pytest.raises(ValueError):
        check_shape(np.zeros((2, 4)), shape_dims)


def test_check_shape_error_shows_actual_and_expected():
    with pytest.raises(ValueError, match=r"expected \(4, 3\), but got \(3, 4\)"):
        check_shape(np.zeros((3, 4)), (4, 3))


def test_check_shape_none():
    with pytest.raises(TypeError, match="cannot be None"):
        check_shape(None, (3, 4))


def test_check_shape_not_an_array():
    with pytest.raises(TypeError, match="must be a NumPy array, got list"):
        check_shape([[1, 2]], (1, 2))


# >> tests for safe_parse_int()

def test_safe_parse_int_valid_string():
    assert safe_parse_int("42") == 42


def test_safe_parse_int_bad_string():
    assert safe_parse_int("abc") is None
    assert safe_parse_int("abc", -1) == -1


@pytest.mark.parametrize("value, expected", [(3.9, 3), (-3.9, -3)])
def test_safe_parse_int_float(value, expected):
    assert safe_parse_int(value) == expected


def test_safe_parse_int_none():
    assert safe_parse_int(None) is None
    assert safe_parse_int(None, -1) == -1


def test_safe_parse_int_empty_string():
    assert safe_parse_int("", -1) == -1


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_safe_parse_int_float_with_no_int_value(value):
    assert safe_parse_int(value, -1) == -1
