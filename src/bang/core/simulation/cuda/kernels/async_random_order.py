# type: ignore

import numba as nb
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

from bang.core.simulation.common import (
    MAX_STATE_SIZE,
    MAX_UPDATE_ORDER_SIZE,
    update_node,
)
from bang.core.simulation.cuda.perturbation import perform_perturbation
from bang.core.simulation.cuda.state_management import update_initial_state


@cuda.jit
def kernel_converge_async_random_order(
    gpu_stateHistory,
    gpu_threadNum,
    gpu_powNum,
    gpu_cumNf,
    gpu_cumCij,
    states,
    nodeNum,
    gpu_perturbation_rate,
    gpu_cumNv,
    gpu_F,
    gpu_varF,
    gpu_initialState,
    gpu_steps,
    gpu_stateSize,
    gpu_extraF,
    gpu_extraFIndex,
    gpu_cumExtraF,
    gpu_extraFCount,
    gpu_extraFIndexCount,
    gpu_npLength,
    gpu_npNode,
    save_history,
):
    idx = cuda.grid(1)

    steps = gpu_steps[0]
    stateSize = gpu_stateSize[0]

    # initialStateCopy = cuda.local.array(shape=(10,), dtype=nb.uint32)
    initialState = cuda.local.array(shape=(MAX_STATE_SIZE,), dtype=nb.uint32)
    update_order = cuda.local.array(shape=(MAX_UPDATE_ORDER_SIZE,), dtype=nb.uint32)

    relative_index = idx * stateSize

    # get initial state of the trajectory this thread will simulate
    # stateSize is the number of 32-bit integers needed to represent one state
    for node_index in range(stateSize):
        initialState[node_index] = gpu_initialState[relative_index + node_index]

    steps = gpu_steps[0]

    for step in range(steps):
        perturbation = False

        perturbation = perform_perturbation(
            gpu_npLength,
            gpu_npNode,
            gpu_perturbation_rate,
            states,
            idx,
            initialState,
        )

        if not perturbation:
            for i in range(nodeNum):
                update_order[i] = i

            for i in range(nodeNum - 1, 0, -1):
                rand = xoroshiro128p_uniform_float32(states, idx)  # Random index
                j = int(rand * (i + 1))

                update_order[i], update_order[j] = update_order[j], update_order[i]

            for i in range(nodeNum):
                node_index = update_order[i]

                index_shift = node_index % 32
                index_state = node_index // 32

                rand = xoroshiro128p_uniform_float32(states, idx)

                update_node(
                    node_index,
                    index_shift,
                    index_state,
                    rand,
                    gpu_cumCij,
                    gpu_cumNf,
                    gpu_cumNv,
                    gpu_F,
                    gpu_extraFIndex,
                    gpu_extraF,
                    gpu_cumExtraF,
                    gpu_varF,
                    gpu_powNum,
                    initialState,
                    initialState,
                )

        update_initial_state(
            gpu_threadNum,
            gpu_stateHistory,
            gpu_initialState,
            stateSize,
            idx,
            step,
            initialState,
            initialState,
            save_history,
        )
