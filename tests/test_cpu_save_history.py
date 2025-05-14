from bang import PBN


def test_save_history_true():
    pbn = PBN(
        2,
        [1, 1],
        [2, 2],
        [[True, True, True, False], [True, False, True, True]],
        [[0, 1], [0, 1]],
        [[1.0], [1.0]],
        0.0,
        [2],
        n_parallel=2,
        update_type="synchronous",
        save_history=True,
    )
    pbn.set_states([[True, False], [False, True]])

    pbn.simple_steps(5, device="cpu")

    assert pbn._history.shape == (7, 2, 1), pbn._history


def test_save_history_no_set_state():
    pbn = PBN(
        2,
        [1, 1],
        [2, 2],
        [[True, True, True, False], [True, False, True, True]],
        [[0, 1], [0, 1]],
        [[1.0], [1.0]],
        0.0,
        [2],
        n_parallel=2,
        update_type="synchronous",
    )

    pbn.simple_steps(5, device="cpu")

    assert pbn._history.shape == (6, 2, 1), pbn._history


def test_save_history_false():
    pbn = PBN(
        2,
        [1, 1],
        [2, 2],
        [[True, True, True, False], [True, False, True, True]],
        [[0, 1], [0, 1]],
        [[1.0], [1.0]],
        0.0,
        [2],
        n_parallel=2,
        update_type="synchronous",
        save_history=False,
    )

    pbn.simple_steps(5, device="cpu")

    assert pbn._history.shape == (1, 2, 1), pbn._history


def test_save_history_false_set_state():
    pbn = PBN(
        2,
        [1, 1],
        [2, 2],
        [[True, True, True, False], [True, False, True, True]],
        [[0, 1], [0, 1]],
        [[1.0], [1.0]],
        0.0,
        [2],
        n_parallel=2,
        update_type="synchronous",
        save_history=False,
    )

    pbn.set_states([[True, False], [False, True]])

    pbn.simple_steps(5, device="cpu")

    assert pbn._history.shape == (2, 2, 1), pbn._history
