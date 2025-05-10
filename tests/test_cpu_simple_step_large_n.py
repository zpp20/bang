import numpy as np
import pytest

from bang import PBN

large_testdata = [
    (32, 1),
    (64, 2),
    (128, 4),
    (256, 8),
    (512, 16),
]


@pytest.mark.parametrize("test_n_nodes, test_n_states", large_testdata)
def test_simple_step_large_nodeset(test_n_nodes, test_n_states):
    pbn_num_func = [1 for _ in range(test_n_nodes)]
    pbn_num_var = [2 for _ in range(test_n_nodes)]
    pbn_probs = [[1.0] for _ in range(test_n_nodes)]
    pbn_var_indexes = [[i // 2, i // 2 + 1] for i in range(test_n_nodes)]
    functions_1 = [[True, True, True, False] for _ in range(test_n_nodes // 2)]
    functions_2 = [[True, False, True, True] for _ in range(test_n_nodes // 2)]
    pbn_functions: list[list[bool]] = []

    for a, b in zip(functions_1, functions_2):
        pbn_functions.append(a)
        pbn_functions.append(b)

    pbn1 = PBN(
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

    pbn1.simple_steps(1, device="cpu")

    last_state = pbn1.last_state

    expected = [[2**32 - 1 for _ in range(test_n_states)]]

    assert np.array_equal(expected, last_state), last_state


missing_pair_testdata = [
    (30, 1),
    (62, 2),
    (126, 4),
    (254, 8),
    (510, 16),
]


@pytest.mark.parametrize("test_n_nodes, test_n_states", missing_pair_testdata)
def test_update_state_large_missing_pair(test_n_nodes, test_n_states):
    # Here we test if updates are correct if the total is not a multiple of 32
    # This way we see if the first state will be less that 2**32 - 1

    pbn_num_func = [1 for _ in range(test_n_nodes)]
    pbn_num_var = [2 for _ in range(test_n_nodes)]
    pbn_probs = [[1.0] for _ in range(test_n_nodes)]
    pbn_var_indexes = [[i // 2, i // 2 + 1] for i in range(test_n_nodes)]
    functions_1 = [[True, True, True, False] for _ in range(test_n_nodes // 2)]
    functions_2 = [[True, False, True, True] for _ in range(test_n_nodes // 2)]
    pbn_functions: list[list[bool]] = []

    for a, b in zip(functions_1, functions_2):
        pbn_functions.append(a)
        pbn_functions.append(b)

    pbn1 = PBN(
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

    pbn1.simple_steps(1, device="cpu")

    last_state = pbn1.last_state

    expected = [[2**32 - 1 for _ in range(test_n_states - 1)] + [2**30 - 1]]

    assert np.array_equal(last_state, expected), last_state
