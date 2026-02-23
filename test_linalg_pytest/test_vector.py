"""
Here are located functions for testing Vector class functionality using pytest, pyunit
"""
import math
import struct
import pytest
from linalg.vector import Vector
from linalg.matrix import Matrix

#pylint: disable=unnecessary-dunder-call


@pytest.mark.parametrize("a, a_len, b, b_len, expected", [
    ([1,2,3], 3, [1,2,3], 3, True),
    ([3,2,1], 3, [1,2,3], 3, False),
    ([1,2,3], 3, [1,2,3], 4, False),
    ([1,2,3,0], 4, [1,2,3], 4, True),
])
def test_eq(a, a_len, b, b_len, expected):
    """
    Tests for equality operator
    """
    assert (Vector(a, a_len) == Vector(b, b_len)) is expected

@pytest.mark.parametrize("a, a_len, b, b_len, expected", [
    ([1,2,3], 3, [1,2,3], 3, False),
    ([3,2,1], 3, [1,2,3], 3, True),
    ([1,2,3], 3, [1,2,3], 4, True),
    ([1,2,3,0], 4, [1,2,3], 4, False),
])
def test_ne(a, a_len, b, b_len, expected):
    """
    Tests for not equality operator
    """
    assert (Vector(a, a_len) != Vector(b, b_len)) is expected

@pytest.mark.parametrize("a, expected_norm", [
    ([1,2,3,4], 5.4772255751),
    ([-1,-2,-3,-4], 5.4772255751),
])
def test_norm(a, expected_norm):
    """
    Tests for norm of the vector
    """
    assert math.isclose(Vector(a, len(a)).norm(), expected_norm)

@pytest.mark.parametrize("a, b, expected", [
    ([1,2], [2,1], False),
    ([1,2], [1,2,3], True),
    ([1,2,4], [1,2,3], False),
])
def test_lt(a, b, expected):
    """
    Tests for less than operator
    """
    assert (Vector(a, len(a)) < Vector(b, len(b))) is expected

@pytest.mark.parametrize("a, b, expected", [
    ([1,2], [2,1], False),
    ([2,3,4], [1,2,3], True),
    ([-1, 4, -3], [1,2,3], True),
    ([2,3,4], [2,3,4.0000000001], False),
])
def test_gt(a, b, expected):
    """
    Tests for grater than operator
    """
    assert (Vector(a, len(a)) > Vector(b, len(b))) is expected

@pytest.mark.parametrize("a, b, expected", [
    ([1,2], [2,1], True),
    ([1.0001, 2.0], [1.0, 2.0001], True),
    ([1.00001, 2.0], [1.0, 2.0001], True),
    ([1.00001, 2.0], [1.0, 2.000001], False),
])
def test_le(a, b, expected):
    """
    Tests for less than or equal operator
    """
    assert (Vector(a, len(a)) <= Vector(b, len(b))) is expected

@pytest.mark.parametrize("a, b, expected", [
    ([1,2], [2,1], True),
    ([1.0001, 2.0], [1.0, 2.0001], False),
    ([1.00001, 2.0], [1.0, 2.0001], False),
    ([1.00001, 2.0], [1.0, 2.000001], True),
])
def test_ge(a, b, expected):
    """
    Tests for greater than or equal operator
    """
    assert (Vector(a, len(a)) >= Vector(b, len(b))) is expected

@pytest.mark.parametrize("a, expected_str", [
    ([2], "Vector [2] has 1 dimention"),
    ([1,2], "Vector [1, 2] has 2 dimentions"),
    ([1,2,3], "Vector [1, 2, 3] has 3 dimentions")
])
def test_str(a, expected_str):
    """
    Tests for conversion to string
    """
    assert str(Vector(a, len(a))) == expected_str

@pytest.mark.parametrize("a, expected", [
    ([1,0,0], True),
    ([0], False),
    ([0,0,0], False)
])
def test_bool(a, expected):
    """
    Tests for bool operator
    """
    assert bool(Vector(a, len(a))) is expected

@pytest.mark.parametrize("a, expected", [
    ([1], 1),
    ([0], 0),
    ([5,], 5),
])
def test_int(a, expected):
    """
    Tests for int operator
    """
    assert int(Vector(a, len(a))) == expected

@pytest.mark.parametrize("a, expected", [
    ([1], 1),
    ([0], 0),
    ([5,], 5),
])
def test_float(a, expected):
    """
    Tests for float operator
    """
    assert int(Vector(a, len(a))) == expected

@pytest.mark.parametrize("a, n_dim", [
    ([1.0, 2.0, 3.0], 3),
    ([1.5, -2.7], 2),
    ([0.0], 1),
    ([1.0, 2.0, 3.0, 4.0, 5.0], 5),
])
def test_bytes_conversion(a, n_dim):
    """
    Tests of __bytes__ method
    """
    vector = Vector(a, n_dim)
    result_bytes = bytes(vector)

    # Verification of length: 4 bytes for dimension + 8 bytes per float
    expected_length = 4 + (8 * n_dim)
    assert len(result_bytes) == expected_length

    # Verification of structure
    dimension_bytes = result_bytes[:4]
    unpacked_dimension = struct.unpack("I", dimension_bytes)[0]
    assert unpacked_dimension == n_dim

def test_bytes_roundtrip():
    """
    Test that we can convert to bytes and reconstruct the vector data
    """
    original_vector = Vector([1.5, -2.7, 3.14], 3)
    vector_bytes = bytes(original_vector)

    # Unpack the dimention bytes manually
    dimension = struct.unpack("I", vector_bytes[:4])[0]

    # Unpack
    values = []
    for i in range(dimension):
        start_idx = 4 + (i * 8)
        end_idx = start_idx + 8
        float_bytes = vector_bytes[start_idx:end_idx]
        value = struct.unpack("d", float_bytes)[0]
        values.append(value)

    assert dimension == 3
    assert values == [1.5, -2.7, 3.14]

@pytest.mark.parametrize("a, expected", [
    ([1,2,3], 3),
    ([1], 1),
])
def test_len(a, expected):
    """
    Tests for __len__ method
    """
    assert len(Vector(a, len(a))) == expected

def test_iterator_basic():
    """
    Test that Vector type is iterable
    """
    test_list = [1,2,3.0]
    assert list(Vector(test_list, len(test_list))) == test_list

def test_iterator_for_loop():
    """
    Test that __next__ method is working
    """
    test_list = [1,2,3.0]
    result_list = []
    for v in Vector(test_list, len(test_list)):
        result_list.append(v)
    assert result_list == test_list

def test_iterator_length_hint():
    """
    Test __lenth_hint___ method
    """
    vector = Vector([1,2,3,4,5], 5)
    iterator = iter(vector)

    assert iterator.__length_hint__() == 5

    next(iterator)

    assert iterator.__length_hint__() == 4

    next(iterator)
    next(iterator)
    next(iterator)
    next(iterator)

    assert iterator.__length_hint__() == 0

@pytest.mark.parametrize("a, start_selector, end_selector, expected", [
    ([1,2,3,4], 0, 2, [1,2]),
    ([1,2,3,4], 1, 2, [2]),
    ([1,2,3,4], 0, 4, [1,2,3,4]),
])
def test_getitem_slice(a, start_selector, end_selector, expected):
    """
    Test for slicing part of the vector
    """
    assert Vector(a, len(a))[start_selector: end_selector] == Vector(expected, len(expected))

@pytest.mark.parametrize("a, index, expected", [
    ([1,2,3,4], 0, 1),
    ([1,2,3,4], 1, 2),
    ([1,2,3,4], 2, 3),
])
def test_getitem_index(a, index, expected):
    """
    Test for indexing part of the vector
    """
    assert Vector(a, len(a))[index] == expected

@pytest.mark.parametrize("a, start_selector, end_selector, b, expected", [
    ([1,2,3,4,5], 1, 3, [6, 7], [1, 6, 7, 4, 5]),
    ([1,2,3,4,5], 0, 4, [6, 7, 8, 9], [6, 7, 8, 9, 5]),
])
def test_setitem_slice(a, start_selector, end_selector, b, expected):
    """
    Test __setitem__ when slice selector
    """
    vector = Vector(a, len(a))
    vector[start_selector: end_selector] = b
    assert vector == Vector(expected, len(expected))

@pytest.mark.parametrize("a, start_selector, end_selector, b", [
    ([1,2,3,4,5], 1, 3, [6, 7, 8]),
    ([1,2,3,4,5], 0, 4, [6, 7]),
])
def test_setitem_except(a, start_selector, end_selector, b):
    """
    Test __setitem when slice selector might change dimention
    """
    with pytest.raises(ValueError):
        Vector(a, len(a))[start_selector: end_selector] = b

@pytest.mark.parametrize("value, test_list, expected", [
    (4, [2,3,4], True),
    (4, [2,3,5], False),
])
def test_contains(value, test_list, expected):
    """
    Test __contains__ dunder method
    """
    assert (value in Vector(test_list, len(test_list))) == expected

@pytest.mark.parametrize("a, b , expected", [
    ([1,2,3], [1,2,3], [2,4,6]),
    ([1.0, 2.5, 3.4], [1.4, 3.2, 7.2], [2.4, 5.7, 10.6]),
])
def test_add_vector(a, b, expected):
    """
    Test add vector to other vector
    """
    assert Vector(a, len(a)) + Vector(b, len(b)) == Vector(expected, len(expected))

@pytest.mark.parametrize("a, b , expected", [
    ([1,2,3], 2, [3,4,5]),
    ([1.0, 2.5, 3.4], 3.1, [4.1, 5.6, 6.5]),
])
def test_add_float(a, b, expected):
    """
    Test add vector to other vector
    """
    assert Vector(a, len(a)) + b == Vector(expected, len(expected))


def test_mul_matrix():
    """
    Test vector * matrix operation
    """
    assert Vector([1,2,3], 3) * Matrix([[1,2,3],[4,5,6],[7,8,9]], 3, 3) == Vector([30, 36, 42], 3)
