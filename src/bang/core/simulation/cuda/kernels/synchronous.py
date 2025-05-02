# type: ignore
import numba as nb
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

from bang.core.simulation.common import MAX_STATE_SIZE, update_node
from bang.core.simulation.cuda.perturbation import perform_perturbation
from bang.core.simulation.cuda.state_management import (
    initialize_state,
    update_initial_state,
)


@cuda.jit
def kernel_converge_sync(
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

    initialStateCopy = cuda.local.array(shape=(MAX_STATE_SIZE,), dtype=nb.uint32)
    initialState = cuda.local.array(shape=(MAX_STATE_SIZE,), dtype=nb.uint32)

    relative_index = idx * stateSize

    initialize_state(
        gpu_initialState,
        stateSize,
        relative_index,
        initialStateCopy,
        initialState,
    )

    steps = gpu_steps[0]

    for step in range(steps):
        perturbation = False
        index_shift = 0
        index_state = 0
        # here, we iterate over intervals of nodes that can be perturbed
        # since npNodee is array of non-perturbable nodes, we iterate over intervals of nodes
        # not containing them;

        # for every interval full of perturbable nodes

        perturbation = perform_perturbation(
            gpu_npLength,
            gpu_npNode,
            gpu_perturbation_rate,
            states,
            idx,
            initialStateCopy,
        )

        if not perturbation:
            for node_index in range(nodeNum):
                rand = xoroshiro128p_uniform_float32(states, idx)

                index_shift = node_index % 32
                index_state = node_index // 32

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
                    initialStateCopy,
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
            initialStateCopy,
            save_history,
        )
