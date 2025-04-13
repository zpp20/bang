# type: ignore

import numba as nb
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


@nb.jit
def initialize_state(
    gpu_initialState, gpu_stateHistory, state_size, relative_index, initialStateCopy, initialState
):
    # get initial state of the trajectory this thread will simulate
    # stateSize is the number of 32-bit integers needed to represent one state
    for node_index in range(state_size):
        initialStateCopy[node_index] = gpu_initialState[relative_index + node_index]
        initialState[node_index] = initialStateCopy[node_index]
        gpu_stateHistory[relative_index + node_index] = initialState[node_index]


@nb.jit
def update_node(
    node_index,
    index_shift,
    indexState,
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
):
    relative_index = 0

    # choose function to update state of node_indexth node
    # we assume that the cumulative probability is very close to 1
    # and rand is (almost) always smaller than it
    while rand > gpu_cumCij[gpu_cumNf[node_index] + relative_index]:
        relative_index += 1

    start = gpu_cumNf[node_index] + relative_index

    element_f = gpu_F[start]
    # number of variables in the function
    start_var_f_index = gpu_cumNv[start]
    result_state_size = gpu_cumNv[start + 1] - start_var_f_index
    shift_num = 0

    # for every variable in the function
    for ind in range(result_state_size):
        relative_index = gpu_varF[start_var_f_index + ind] // 32
        # extract 32-bit integer containing the variable
        state_fragment = initialStateCopy[relative_index]

        if ((state_fragment >> (gpu_varF[start_var_f_index + ind]) % 32) & 1) != 0:
            shift_num += gpu_powNum[1][ind]

    # if we have more than 5 variables, we need to use our extraF array
    if shift_num > 32:
        tt = 0

        while gpu_extraFIndex[tt] != start:
            tt += 1

        element_f = gpu_extraF[gpu_cumExtraF[tt] + ((shift_num - 32) // 32)]
        shift_num = shift_num % 32

    element_f = element_f >> shift_num

    initialState[indexState] ^= (-(element_f & 1) ^ initialStateCopy[indexState]) & (
        1 << (node_index - indexState * 32)
    )

    index_shift += 1


@nb.jit
def update_initial_state(
    gpu_threadNum,
    gpu_stateHistory,
    gpu_initialState,
    stateSize,
    idx,
    step,
    initialState,
    initialStateCopy,
):
    relative_index = stateSize * idx

    for node_index in range(stateSize):
        initialStateCopy[node_index] = initialState[node_index]
        gpu_initialState[relative_index + node_index] = initialStateCopy[node_index]

        gpu_stateHistory[
            (step + 1) * gpu_threadNum[0] + relative_index + node_index
        ] = initialStateCopy[node_index]


@cuda.jit(device=True)
def perform_perturbation(
    gpu_npLength,
    gpu_npNode,
    gpu_perturbation_rate,
    states,
    idx,
    initialStateCopy,
):
    indexState = 0
    start = 0
    perturbation = False

    # here, we iterate over intervals of nodes that can be perturbed
    # since npNodee is array of non-perturbable nodes, we iterate over intervals of nodes
    # not containing them;
    for t in range(gpu_npLength[0]):
        # for every perturbable node in the interval
        for node_index in range(start, gpu_npNode[t]):
            if xoroshiro128p_uniform_float32(states, idx) < gpu_perturbation_rate[0]:
                perturbation = True
                indexState = node_index // 32
                initialStateCopy[indexState] ^= 1 << (node_index % 32)

        start = gpu_npNode[t] + 1

    return perturbation


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
):
    idx = cuda.grid(1)

    steps = gpu_steps[0]
    stateSize = gpu_stateSize[0]

    initialStateCopy = cuda.local.array(shape=(10,), dtype=nb.int32)
    initialState = cuda.local.array(shape=(4,), dtype=nb.int32)

    relative_index = idx * stateSize

    initialize_state(
        gpu_initialState,
        gpu_stateHistory,
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

                index_shift = 0

                if index_shift == 32:
                    index_state += 1
                    index_shift = 0

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

                index_shift += 1

        update_initial_state(
            gpu_threadNum,
            gpu_stateHistory,
            gpu_initialState,
            stateSize,
            idx,
            step,
            initialState,
            initialStateCopy,
        )


@cuda.jit
def kernel_converge_async(
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
):
    idx = cuda.grid(1)

    steps = gpu_steps[0]
    stateSize = gpu_stateSize[0]

    # initialStateCopy = cuda.local.array(shape=(10,), dtype=nb.int32)
    initialState = cuda.local.array(shape=(4,), dtype=nb.int32)
    update_order = cuda.local.array(shape=(400,), dtype=nb.int32)

    relative_index = idx * stateSize

    # get initial state of the trajectory this thread will simulate
    # stateSize is the number of 32-bit integers needed to represent one state
    for node_index in range(stateSize):
        initialState[node_index] = gpu_initialState[relative_index + node_index]
        gpu_stateHistory[relative_index + node_index] = initialState[node_index]

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
        )
