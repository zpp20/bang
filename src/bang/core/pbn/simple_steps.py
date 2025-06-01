import datetime
import typing

import numba
import numpy as np
import numpy.typing as npt
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

if typing.TYPE_CHECKING:
    from bang.core import PBN

from bang.core.pbn.array_management import convert_pbn_to_ndarrays
from bang.core.simulation.cpu import (
    cpu_converge_async_one_random,
    cpu_converge_async_random_order,
    cpu_converge_sync,
)
from bang.core.simulation.cuda import (
    kernel_converge_async_one_random,
    kernel_converge_async_random_order,
    kernel_converge_sync,
)


def invoke_cuda_simulation(
    pbn: "PBN",
    n_steps: int,
    actions: npt.NDArray[np.uint] | None = None,
    stream=None,
):
    if stream is None:
        stream = cuda.default_stream()

    if pbn._latest_state is None or pbn._history is None:
        raise ValueError("Initial state must be set before simulation")

    # This could happen when GPU is not available when PBN is created, but becomes available afterwards
    if pbn._gpu_memory_container is None:
        pbn._create_memory_container()

    assert pbn._gpu_memory_container is not None

    if actions is not None:
        pbn._latest_state = pbn._perturb_state_by_actions(actions, pbn._latest_state)
        pbn._history = np.concatenate([pbn._history, pbn._latest_state], axis=0)

    pbn._gpu_memory_container.gpu_initialState.copy_to_device(
        pbn._latest_state.reshape(pbn._n_parallel * pbn.state_size)
    )
    pbn._gpu_memory_container.gpu_steps.copy_to_device(np.array([n_steps], dtype=np.uint32))

    states = create_xoroshiro128p_states(
        pbn._n_parallel, seed=numba.uint64(datetime.datetime.now().timestamp())
    )

    block = pbn._n_parallel // 32

    if block == 0:
        block = 1

    blockSize = 32

    if pbn.update_type == "asynchronous_one_random":
        kernel_converge_async_one_random[block, blockSize, stream](  # type: ignore
            pbn._gpu_memory_container.gpu_stateHistory,
            pbn._gpu_memory_container.gpu_threadNum,
            pbn._gpu_memory_container.gpu_powNum,
            pbn._gpu_memory_container.gpu_cumNf,
            pbn._gpu_memory_container.gpu_cumCij,
            states,
            pbn.n_nodes,
            pbn._gpu_memory_container.gpu_perturbation_rate,
            pbn._gpu_memory_container.gpu_cumNv,
            pbn._gpu_memory_container.gpu_F,
            pbn._gpu_memory_container.gpu_varF,
            pbn._gpu_memory_container.gpu_initialState,
            pbn._gpu_memory_container.gpu_steps,
            pbn._gpu_memory_container.gpu_stateSize,
            pbn._gpu_memory_container.gpu_extraF,
            pbn._gpu_memory_container.gpu_extraFIndex,
            pbn._gpu_memory_container.gpu_cumExtraF,
            pbn._gpu_memory_container.gpu_extraFCount,
            pbn._gpu_memory_container.gpu_extraFIndexCount,
            pbn._gpu_memory_container.gpu_npLength,
            pbn._gpu_memory_container.gpu_npNode,
            pbn.save_history,
        )
    elif pbn.update_type == "asynchronous_random_order":
        kernel_converge_async_random_order[block, blockSize, stream](  # type: ignore
            pbn._gpu_memory_container.gpu_stateHistory,
            pbn._gpu_memory_container.gpu_threadNum,
            pbn._gpu_memory_container.gpu_powNum,
            pbn._gpu_memory_container.gpu_cumNf,
            pbn._gpu_memory_container.gpu_cumCij,
            states,
            pbn.n_nodes,
            pbn._gpu_memory_container.gpu_perturbation_rate,
            pbn._gpu_memory_container.gpu_cumNv,
            pbn._gpu_memory_container.gpu_F,
            pbn._gpu_memory_container.gpu_varF,
            pbn._gpu_memory_container.gpu_initialState,
            pbn._gpu_memory_container.gpu_steps,
            pbn._gpu_memory_container.gpu_stateSize,
            pbn._gpu_memory_container.gpu_extraF,
            pbn._gpu_memory_container.gpu_extraFIndex,
            pbn._gpu_memory_container.gpu_cumExtraF,
            pbn._gpu_memory_container.gpu_extraFCount,
            pbn._gpu_memory_container.gpu_extraFIndexCount,
            pbn._gpu_memory_container.gpu_npLength,
            pbn._gpu_memory_container.gpu_npNode,
            pbn.save_history,
        )
    elif pbn.update_type == "synchronous":
        kernel_converge_sync[block, blockSize, stream](  # type: ignore
            pbn._gpu_memory_container.gpu_stateHistory,
            pbn._gpu_memory_container.gpu_threadNum,
            pbn._gpu_memory_container.gpu_powNum,
            pbn._gpu_memory_container.gpu_cumNf,
            pbn._gpu_memory_container.gpu_cumCij,
            states,
            pbn.n_nodes,
            pbn._gpu_memory_container.gpu_perturbation_rate,
            pbn._gpu_memory_container.gpu_cumNv,
            pbn._gpu_memory_container.gpu_F,
            pbn._gpu_memory_container.gpu_varF,
            pbn._gpu_memory_container.gpu_initialState,
            pbn._gpu_memory_container.gpu_steps,
            pbn._gpu_memory_container.gpu_stateSize,
            pbn._gpu_memory_container.gpu_extraF,
            pbn._gpu_memory_container.gpu_extraFIndex,
            pbn._gpu_memory_container.gpu_cumExtraF,
            pbn._gpu_memory_container.gpu_extraFCount,
            pbn._gpu_memory_container.gpu_extraFIndexCount,
            pbn._gpu_memory_container.gpu_npLength,
            pbn._gpu_memory_container.gpu_npNode,
            pbn.save_history,
        )
    else:
        raise ValueError(f"Unsupported update type: {pbn.update_type}")

    cuda.synchronize()

    last_state = pbn._gpu_memory_container.gpu_initialState.copy_to_host()
    run_history = pbn._gpu_memory_container.gpu_stateHistory.copy_to_host()

    pbn._latest_state = last_state.reshape((pbn._n_parallel, pbn.state_size))

    if pbn.save_history:
        run_history = run_history.reshape((-1, pbn._n_parallel, pbn.state_size))[
            : n_steps + 1, :, :
        ]

        if pbn._history is not None:
            pbn._history = np.concatenate([pbn._history, run_history[1:, :, :]], axis=0)
        else:
            pbn.history = run_history


def invoke_cpu_simulation(pbn: "PBN", n_steps: int, actions: npt.NDArray[np.uint] | None = None):
    """
    Simulates the PBN for a given number of steps using CPU-based kernels.

    :param n_steps: Number of steps to simulate.
    :type n_steps: int
    :param actions: Array of actions to be performed on the PBN. Defaults to None.
    :type actions: npt.NDArray[np.uint], optional
    :raises ValueError: If the initial state is not set before simulation.
    """
    if pbn._latest_state is None or pbn._history is None:
        raise ValueError("Initial state must be set before simulation")

    if actions is not None:
        pbn._latest_state = pbn._perturb_state_by_actions(actions, pbn._latest_state)
        pbn._history = np.concatenate([pbn._history, pbn._latest_state], axis=0)

    # Convert PBN data to numpy arrays
    pbn_data = convert_pbn_to_ndarrays(pbn, n_steps, pbn.save_history)

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
    ) = pbn_data

    # Select the appropriate CPU kernel based on the update type
    if pbn.update_type == "asynchronous_random_order":
        cpu_converge_async_random_order(
            state_history,
            thread_num,
            pow_num,
            cum_function_count,
            function_probabilities,
            pbn.n_nodes,
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
            non_perturbed_count,
            perturbation_blacklist,
            pbn.save_history,
        )
    elif pbn.update_type == "synchronous":
        cpu_converge_sync(
            state_history,
            thread_num,
            pow_num,
            cum_function_count,
            function_probabilities,
            pbn.n_nodes,
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
            non_perturbed_count,
            perturbation_blacklist,
            pbn.save_history,
        )
    elif pbn.update_type == "asynchronous_one_random":
        cpu_converge_async_one_random(
            state_history,
            thread_num,
            pow_num,
            cum_function_count,
            function_probabilities,
            pbn.n_nodes,
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
            non_perturbed_count,
            perturbation_blacklist,
            pbn.save_history,
        )
    else:
        raise ValueError(f"Unsupported update type: {pbn.update_type}")

    # Reshape and update the state and history
    last_state = initial_state.reshape((pbn._n_parallel, pbn.state_size))
    pbn._latest_state = last_state

    if pbn.save_history:
        run_history = state_history.reshape((-1, pbn._n_parallel, pbn.state_size))[
            : n_steps + 1, :, :
        ]

        if pbn._history is not None:
            pbn._history = np.concatenate([pbn._history, run_history[1:, :, :]], axis=0)
        else:
            pbn.history = run_history
