import numpy as np
import numpy.typing as npt

import datetime

import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

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

from bang.core.pbn.array_management import convert_pbn_to_ndarrays

from bang.core import PBN


def invoke_cuda_simulation(pbn: "PBN", n_steps: int, actions: npt.NDArray[np.uint] | None = None):
    if pbn.latest_state is None or pbn.history is None:
        raise ValueError("Initial state must be set before simulation")

    # This could happen when GPU is not available when PBN is created, but becomes available afterwards
    if pbn.gpu_memory_container is None:
        pbn._create_memory_container()

    assert pbn.gpu_memory_container is not None

    if actions is not None:
        pbn.latest_state = pbn._perturb_state_by_actions(actions, pbn.latest_state)
        pbn.history = np.concatenate([pbn.history, pbn.latest_state], axis=0)

    pbn.gpu_memory_container.gpu_initialState.copy_to_device(
        pbn.latest_state.reshape(pbn.n_parallel * pbn.stateSize())
    )
    pbn.gpu_memory_container.gpu_steps.copy_to_device(np.array([n_steps], dtype=np.uint32))

    states = create_xoroshiro128p_states(
        pbn.n_parallel, seed=numba.uint64(datetime.datetime.now().timestamp())
    )

    block = pbn.n_parallel // 32

    if block == 0:
        block = 1

    blockSize = 32

    if pbn.update_type == "asynchronous_one_random":
        kernel_converge_async_one_random[block, blockSize](  # type: ignore
            pbn.gpu_memory_container.gpu_stateHistory,
            pbn.gpu_memory_container.gpu_threadNum,
            pbn.gpu_memory_container.gpu_powNum,
            pbn.gpu_memory_container.gpu_cumNf,
            pbn.gpu_memory_container.gpu_cumCij,
            states,
            pbn.getN(),
            pbn.gpu_memory_container.gpu_perturbation_rate,
            pbn.gpu_memory_container.gpu_cumNv,
            pbn.gpu_memory_container.gpu_F,
            pbn.gpu_memory_container.gpu_varF,
            pbn.gpu_memory_container.gpu_initialState,
            pbn.gpu_memory_container.gpu_steps,
            pbn.gpu_memory_container.gpu_stateSize,
            pbn.gpu_memory_container.gpu_extraF,
            pbn.gpu_memory_container.gpu_extraFIndex,
            pbn.gpu_memory_container.gpu_cumExtraF,
            pbn.gpu_memory_container.gpu_extraFCount,
            pbn.gpu_memory_container.gpu_extraFIndexCount,
            pbn.gpu_memory_container.gpu_npLength,
            pbn.gpu_memory_container.gpu_npNode,
            pbn.save_history,
        )
    elif pbn.update_type == "asynchronous_random_order":
        kernel_converge_async_random_order[block, blockSize](  # type: ignore
            pbn.gpu_memory_container.gpu_stateHistory,
            pbn.gpu_memory_container.gpu_threadNum,
            pbn.gpu_memory_container.gpu_powNum,
            pbn.gpu_memory_container.gpu_cumNf,
            pbn.gpu_memory_container.gpu_cumCij,
            states,
            pbn.getN(),
            pbn.gpu_memory_container.gpu_perturbation_rate,
            pbn.gpu_memory_container.gpu_cumNv,
            pbn.gpu_memory_container.gpu_F,
            pbn.gpu_memory_container.gpu_varF,
            pbn.gpu_memory_container.gpu_initialState,
            pbn.gpu_memory_container.gpu_steps,
            pbn.gpu_memory_container.gpu_stateSize,
            pbn.gpu_memory_container.gpu_extraF,
            pbn.gpu_memory_container.gpu_extraFIndex,
            pbn.gpu_memory_container.gpu_cumExtraF,
            pbn.gpu_memory_container.gpu_extraFCount,
            pbn.gpu_memory_container.gpu_extraFIndexCount,
            pbn.gpu_memory_container.gpu_npLength,
            pbn.gpu_memory_container.gpu_npNode,
            pbn.save_history,
        )
    elif pbn.update_type == "synchronous":
        kernel_converge_sync[block, blockSize](  # type: ignore
            pbn.gpu_memory_container.gpu_stateHistory,
            pbn.gpu_memory_container.gpu_threadNum,
            pbn.gpu_memory_container.gpu_powNum,
            pbn.gpu_memory_container.gpu_cumNf,
            pbn.gpu_memory_container.gpu_cumCij,
            states,
            pbn.getN(),
            pbn.gpu_memory_container.gpu_perturbation_rate,
            pbn.gpu_memory_container.gpu_cumNv,
            pbn.gpu_memory_container.gpu_F,
            pbn.gpu_memory_container.gpu_varF,
            pbn.gpu_memory_container.gpu_initialState,
            pbn.gpu_memory_container.gpu_steps,
            pbn.gpu_memory_container.gpu_stateSize,
            pbn.gpu_memory_container.gpu_extraF,
            pbn.gpu_memory_container.gpu_extraFIndex,
            pbn.gpu_memory_container.gpu_cumExtraF,
            pbn.gpu_memory_container.gpu_extraFCount,
            pbn.gpu_memory_container.gpu_extraFIndexCount,
            pbn.gpu_memory_container.gpu_npLength,
            pbn.gpu_memory_container.gpu_npNode,
            pbn.save_history,
        )
    else:
        raise ValueError(f"Unsupported update type: {pbn.update_type}")

    cuda.synchronize()

    last_state = pbn.gpu_memory_container.gpu_initialState.copy_to_host()
    run_history = pbn.gpu_memory_container.gpu_stateHistory.copy_to_host()

    pbn.latest_state = last_state.reshape((pbn.n_parallel, pbn.stateSize()))

    if pbn.save_history:
        run_history = run_history.reshape((-1, pbn.n_parallel, pbn.stateSize()))[
            : n_steps + 1, :, :
        ]

        if pbn.history is not None:
            pbn.history = np.concatenate([pbn.history, run_history[1:, :, :]], axis=0)
        else:
            pbn.history = run_history

def invoke_cpu_simulation(self, n_steps: int, actions: npt.NDArray[np.uint] | None = None):
    """
    Simulates the PBN for a given number of steps using CPU-based kernels.

    :param n_steps: Number of steps to simulate.
    :type n_steps: int
    :param actions: Array of actions to be performed on the PBN. Defaults to None.
    :type actions: npt.NDArray[np.uint], optional
    :raises ValueError: If the initial state is not set before simulation.
    """
    if self.latest_state is None or self.history is None:
        raise ValueError("Initial state must be set before simulation")

    if actions is not None:
        self.latest_state = self._perturb_state_by_actions(actions, self.latest_state)
        self.history = np.concatenate([self.history, self.latest_state], axis=0)

    # Convert PBN data to numpy arrays
    pbn_data = convert_pbn_to_ndarrays(self, n_steps)

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

    if self.update_type == "asynchronous_random_order":
        cpu_converge_async_random_order(
            state_history,
            thread_num,
            pow_num,
            cum_function_count,
            function_probabilities,
            self.getN(),
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
            self.save_history,
        )
    elif self.update_type == "synchronous":
        cpu_converge_sync(
            state_history,
            thread_num,
            pow_num,
            cum_function_count,
            function_probabilities,
            self.getN(),
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
            self.save_history,
        )
    elif self.update_type == "asynchronous_one_random":
        cpu_converge_async_one_random(
            state_history,
            thread_num,
            pow_num,
            cum_function_count,
            function_probabilities,
            self.getN(),
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
            self.save_history,
        )
    else:
        raise ValueError(f"Unsupported update type: {self.update_type}")

    # Reshape and update the state and history
    last_state = initial_state.reshape((self.n_parallel, self.stateSize()))
    run_history = state_history.reshape((n_steps + 1, self.n_parallel, self.stateSize()))

    self.latest_state = last_state

    if self.history is not None:
        self.history = np.concatenate([self.history, run_history[1:, :, :]], axis=0)
    else:
        self.history = run_history