import numpy as np
import pytest

from bang.core import PBN


def test_should_convert_short_list_of_bools_to_state():
    state = PBN._bools_to_state_array([True, False], 2)

    assert np.array_equal(state, np.array([1]))


def test_should_raise_on_invalid_bool_count():
    with pytest.raises(ValueError):
        _ = PBN._bools_to_state_array([True, False], 4)


def test_should_convert_long_list_of_bools_to_state():
    state = PBN._bools_to_state_array([False for _ in range(33)], 33)

    assert np.array_equal(state, np.array([0, 0]))
