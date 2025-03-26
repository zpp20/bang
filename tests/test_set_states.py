import numpy as np

from bang import PBN


def test_set_states_same_n_parallel():
    pbn = PBN(
        2,
        [1, 1],
        [1, 1],
        [[True, False], [False, True]],
        [[1], [0]],
        [[1.0], [1.0]],
        0.01,
        [2],
        n_parallel=1,
    )

    pbn.set_states([[False, False]])

    assert pbn.get_trajectories().shape == (2, 1, 1)
    assert pbn.latest_state.shape == (1, 1)

    assert np.array_equal(pbn.latest_state, [[0]])
    assert np.array_equal(pbn.get_trajectories(), [[[0]], [[0]]])


def test_set_states_different_n_parallel():
    pbn = PBN(
        2,
        [1, 1],
        [1, 1],
        [[True, False], [False, True]],
        [[1], [0]],
        [[1.0], [1.0]],
        0.01,
        [2],
        n_parallel=1,
    )

    pbn.set_states([[False, False], [False, True]])

    assert pbn.get_trajectories().shape == (1, 2, 1)
    assert pbn.latest_state.shape == (2, 1)

    assert np.array_equal(pbn.latest_state, [[0], [2]])
    assert np.array_equal(pbn.get_trajectories(), [[[0], [2]]])


def test_set_states_multiple_calls():
    pbn = PBN(
        2,
        [1, 1],
        [1, 1],
        [[True, False], [False, True]],
        [[1], [0]],
        [[1.0], [1.0]],
        0.01,
        [2],
        n_parallel=1,
    )

    pbn.set_states([[True, False]])
    pbn.set_states([[False, False]])
    pbn.set_states([[False, True]])

    assert pbn.get_trajectories().shape == (4, 1, 1)
    assert pbn.latest_state.shape == (1, 1)

    assert np.array_equal(pbn.latest_state, [[2]])
    assert np.array_equal(pbn.get_trajectories(), [[[0]], [[1]], [[0]], [[2]]])
