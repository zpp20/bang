from bang.core.PBN import PBN


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

    pbn.simple_steps_cpu(5)

    assert pbn.history.shape == (7, 2, 1), pbn.history


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

    pbn.simple_steps_cpu(5)

    assert pbn.history.shape == (6, 2, 1), pbn.history


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

    pbn.simple_steps_cpu(5)

    assert pbn.history.shape == (1, 2, 1), pbn.history


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

    pbn.simple_steps_cpu(5)

    assert pbn.history.shape == (2, 2, 1), pbn.history
