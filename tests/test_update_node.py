import numpy as np
import pytest

from bang.core.cuda.simulation import update_node
from bang.core.PBN import PBN


def test_update_state_simple():
    pbn1 = PBN(
        2,
        [1, 1],
        [2, 2],
        [[True, True, True, False], [True, False, True, True]],
        [[0, 1], [0, 1]],
        [[1.0], [1.0]],
        0.0,
        [2],
        n_parallel=1,
    )

    prepared_data = pbn1.pbn_data_to_np_arrays(1)

    (
        _,
        _,
        pow_num,
        cum_function_count,
        function_probabilities,
        _,
        cum_variable_count,
        functions,
        function_variables,
        initial_state,
        _,
        _,
        extra_functions,
        extra_functions_index,
        cum_extra_functions,
        _,
        _,
        _,
        _,
    ) = prepared_data

    initial_state = [0]
    initial_state_copy = initial_state.copy()

    for i in range(2):
        update_node(
            i,
            0,
            0,
            0.0,
            function_probabilities,
            cum_function_count,
            cum_variable_count,
            functions,
            extra_functions_index,
            extra_functions,
            cum_extra_functions,
            function_variables,
            pow_num,
            initial_state_copy,
            initial_state,
        )

    assert np.array_equal(initial_state, [3])


large_testdata = [
    (32, 1),
    (64, 2),
    (128, 4),
    (256, 8),
    (512, 16),
]


@pytest.mark.parametrize("test_n_nodes, test_n_states", large_testdata)
def test_update_state_large(test_n_nodes, test_n_states):
    # Integration test for the update_node function given a large PBN
    # with 256 nodes and 2 functions per node.
    # Some of the capabilities executed on the GPU are mocked, like the behaviour
    # of calculating indexShift and indexState.

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
    )

    prepared_data = pbn1.pbn_data_to_np_arrays(1)

    (
        _,
        _,
        pow_num,
        cum_function_count,
        function_probabilities,
        _,
        cum_variable_count,
        functions,
        function_variables,
        initial_state,
        _,
        _,
        extra_functions,
        extra_functions_index,
        cum_extra_functions,
        _,
        _,
        _,
        _,
    ) = prepared_data

    initial_state = [0 for _ in range(test_n_states)]
    initial_state_copy = initial_state.copy()

    for i in range(test_n_nodes):
        update_node(
            i,
            i % 32,
            i // 32,
            0.0,
            function_probabilities,
            cum_function_count,
            cum_variable_count,
            functions,
            extra_functions_index,
            extra_functions,
            cum_extra_functions,
            function_variables,
            pow_num,
            initial_state_copy,
            initial_state,
        )

    assert np.array_equal(initial_state, [2**32 - 1 for _ in range(test_n_states)]), initial_state


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
    )

    prepared_data = pbn1.pbn_data_to_np_arrays(1)

    (
        _,
        _,
        pow_num,
        cum_function_count,
        function_probabilities,
        _,
        cum_variable_count,
        functions,
        function_variables,
        initial_state,
        _,
        _,
        extra_functions,
        extra_functions_index,
        cum_extra_functions,
        _,
        _,
        _,
        _,
    ) = prepared_data

    initial_state = [0 for _ in range(test_n_states)]
    initial_state_copy = initial_state.copy()

    for i in range(test_n_nodes):
        update_node(
            i,
            i % 32,
            i // 32,
            0.0,
            function_probabilities,
            cum_function_count,
            cum_variable_count,
            functions,
            extra_functions_index,
            extra_functions,
            cum_extra_functions,
            function_variables,
            pow_num,
            initial_state_copy,
            initial_state,
        )

    expected = [2**32 - 1 for _ in range(test_n_states - 1)] + [2**30 - 1]

    assert np.array_equal(initial_state, expected), initial_state


def test_update_node_permutations():
    # Test if a different order of updates gives the same result for a fixed point
    # - this is important for asynchronous updates
    pbn1 = PBN(
        2,
        [1, 1],
        [2, 2],
        [[True, True, True, False], [True, False, True, True]],
        [[0, 1], [0, 1]],
        [[1.0], [1.0]],
        0.0,
        [2],
        n_parallel=1,
        update_type_int=0,
    )

    prepared_data = pbn1.pbn_data_to_np_arrays(1)

    (
        _,
        _,
        pow_num,
        cum_function_count,
        function_probabilities,
        _,
        cum_variable_count,
        functions,
        function_variables,
        initial_state,
        _,
        _,
        extra_functions,
        extra_functions_index,
        cum_extra_functions,
        _,
        _,
        _,
        _,
    ) = prepared_data

    initial_state = [1]
    initial_state_2 = [1]

    indexShift = 0
    indexState = 0
    perturbation = 0.0

    for i in range(2):
        update_node(
            i,
            indexShift,
            indexState,
            perturbation,
            function_probabilities,
            cum_function_count,
            cum_variable_count,
            functions,
            extra_functions_index,
            extra_functions,
            cum_extra_functions,
            function_variables,
            pow_num,
            initial_state,
            initial_state,
        )

    for i in range(2)[::-1]:
        update_node(
            i,
            indexShift,
            indexState,
            perturbation,
            function_probabilities,
            cum_function_count,
            cum_variable_count,
            functions,
            extra_functions_index,
            extra_functions,
            cum_extra_functions,
            function_variables,
            pow_num,
            initial_state_2,
            initial_state_2,
        )

    assert np.array_equal(initial_state, [1]), initial_state
    assert np.array_equal(initial_state_2, [1]), initial_state_2
