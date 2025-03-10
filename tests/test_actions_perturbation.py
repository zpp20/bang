import numpy as np

from bang.core.PBN import PBN


def test_should_perturb_at_index():
    result = PBN._perturb_state_by_actions(np.array([0]), np.array([[0]]))

    assert np.array_equal(result, np.array([[1]]))


def test_should_perturb_at_multiple_indices():
    result = PBN._perturb_state_by_actions(np.array([0, 1, 2]), np.array([[0]]))

    assert np.array_equal(result, np.array([[7]]))


def test_should_perturb_in_multiple_trajectories():
    result = PBN._perturb_state_by_actions(np.array([0, 1, 2]), np.array([[0], [1]]))

    assert np.array_equal(result, np.array([[7], [6]]))
