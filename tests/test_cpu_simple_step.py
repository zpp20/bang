import numpy as np
import pytest

from bang import PBN


def test_fixpoint_async_step():
    pbn1 = PBN(
        2,
        [1, 1],
        [2, 2],
        [[True, True, True, False], [True, False, True, True]],
        [[0, 1], [0, 1]],
        [[1.0], [1.0]],
        0.0,
        [2],
        n_parallel=2,
        update_type="asynchronous_random_order",
    )
    pbn1.set_states([[True, False], [False, False]])

    pbn1.simple_steps(101, device="cpu")

    assert np.array_equal([[1], [3]], pbn1._latest_state), pbn1._latest_state


def test_fixpoint_sync_step():
    pbn1 = PBN(
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
    pbn1.set_states([[True, False], [True, False]])

    pbn1.simple_steps(21, device="cpu")

    assert np.array_equal([[1], [1]], pbn1._latest_state), pbn1._latest_state


def test_independent_pair_sync_step():
    test_n_nodes = 4

    pbn_num_func = [1 for _ in range(test_n_nodes)]
    pbn_num_var = [2 for _ in range(test_n_nodes)]
    pbn_probs = [[1.0] for _ in range(test_n_nodes)]
    pbn_var_indexes = [[(i // 2) * 2, (i // 2) * 2 + 1] for i in range(test_n_nodes)]
    functions_1 = [[True, True, True, False] for _ in range(test_n_nodes // 2)]
    functions_2 = [[True, False, True, True] for _ in range(test_n_nodes // 2)]
    pbn_functions: list[list[bool]] = []

    for a, b in zip(functions_1, functions_2):
        pbn_functions.append(a)
        pbn_functions.append(b)

    pbn2 = PBN(
        test_n_nodes,
        pbn_num_func,
        pbn_num_var,
        pbn_functions,
        pbn_var_indexes,
        pbn_probs,
        0.0,
        [test_n_nodes],
        n_parallel=1,
        update_type="synchronous",
    )

    pbn2.simple_steps(1, device="cpu")
    assert np.array_equal([[15]], pbn2._latest_state)

    pbn2.simple_steps(1, device="cpu")
    assert np.array_equal([[10]], pbn2._latest_state)

    pbn2.simple_steps(1, device="cpu")
    assert np.array_equal([[15]], pbn2._latest_state)


@pytest.mark.parametrize("n_parallel", [16, 32, 64, 128, 256, 512])
def test_large_n_parallel_sync(n_parallel):
    pbn1 = PBN(
        2,
        [1, 1],
        [2, 2],
        [[True, True, True, False], [True, False, True, True]],
        [[0, 1], [0, 1]],
        [[1.0], [1.0]],
        0.0,
        [2],
        n_parallel,
        update_type="synchronous",
    )
    pbn1.set_states([[False, False] for _ in range(n_parallel)])

    pbn1.simple_steps(21, device="cpu")

    assert np.array_equal([[3] for _ in range(n_parallel)], pbn1._latest_state), pbn1._latest_state


@pytest.mark.parametrize("n_parallel", [16, 32, 64, 128, 256, 512])
def test_large_n_parallel_async(n_parallel):
    pbn1 = PBN(
        2,
        [1, 1],
        [2, 2],
        [[True, True, True, False], [True, False, True, True]],
        [[0, 1], [0, 1]],
        [[1.0], [1.0]],
        0.0,
        [2],
        n_parallel,
        update_type="asynchronous_random_order",
    )
    pbn1.set_states([[False, False] for _ in range(n_parallel)])

    pbn1.simple_steps(21, device="cpu")

    assert np.array_equal([[3] for _ in range(n_parallel)], pbn1._latest_state), pbn1._latest_state
