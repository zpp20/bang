import numpy as np
import pytest

from bang.core.pbn.utils.state_printing import (
    convert_from_binary_representation,
    convert_to_binary_representation,
)


def test_array_of_ints():
    state = [4, 5, 6, 7]

    output = convert_to_binary_representation(state, bit_length=3)

    assert np.array_equal(
        [[False, False, True], [True, False, True], [False, True, True], [True, True, True]], output
    ), output


def test_array_of_floats():
    state = [4.0, 5.0, 6.0, 7.0]

    output = convert_to_binary_representation(state, bit_length=3)

    assert np.array_equal(
        [[False, False, True], [True, False, True], [False, True, True], [True, True, True]], output
    ), output


def test_array_of_numpy_ints():
    state = np.array([4, 5, 6, 7], dtype=np.int32)

    output = convert_to_binary_representation(state, bit_length=3)

    assert np.array_equal(
        [[False, False, True], [True, False, True], [False, True, True], [True, True, True]], output
    ), output


def test_negative_value():
    state = [4, -5, 6, 7]

    with pytest.raises(ValueError):
        convert_to_binary_representation(state, bit_length=3)


def test_non_integer_floats():
    state = [4, 5, 6, 7.5]

    with pytest.raises(ValueError):
        convert_to_binary_representation(state, bit_length=3)


def test_example_attractor_output():
    attractors = np.array([[np.uint32(0), np.uint32(1)], [np.uint32(2), np.uint32(3)]])

    output = convert_to_binary_representation(attractors, bit_length=2)

    assert np.array_equal(
        [[[False, False], [True, False]], [[False, True], [True, True]]], output
    ), output


def test_coversion_to_array_of_ints():
    state = [[False, False, True], [True, False, True], [False, True, True], [True, True, True]]

    output = convert_from_binary_representation(state)

    assert np.array_equal([4, 5, 6, 7], output), output


def test_coversion_to_array_of_nested_ints():
    attractors = [[[False, False], [True, False]], [[False, True], [True, True]]]

    output = convert_from_binary_representation(attractors)

    assert np.array_equal([[0, 1], [2, 3]], output), output
