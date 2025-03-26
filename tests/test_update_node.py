import numpy as np

from bang.core.cuda.simulation import update_node
from bang.core.PBN import PBN


def test_update_simple_node():
    initial_state = [0]

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
        state_history,
        thread_num,
        pow_num,
        cum_function_count,
        function_probabilities,
        perturbation_rate,
        cum_variable_count,
        functions,
        function_variables,
        initial_state,
        steps,
        state_size,
        extra_functions,
        extra_functions_index,
        cum_extra_functions,
        extra_function_count,
        extra_function_index_count,
        perturbation_blacklist,
        non_perturbed_count,
    ) = prepared_data

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
