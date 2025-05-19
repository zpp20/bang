import numba as nb


@nb.jit
def initialize_state(gpu_initialState, state_size, relative_index, initialStateCopy, initialState):
    # get initial state of the trajectory this thread will simulate
    # stateSize is the number of 32-bit integers needed to represent one state
    for node_index in range(state_size):
        initialStateCopy[node_index] = gpu_initialState[relative_index + node_index]
        initialState[node_index] = initialStateCopy[node_index]


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
    save_history,
):
    relative_index = stateSize * idx

    for node_index in range(stateSize):
        initialStateCopy[node_index] = initialState[node_index]
        gpu_initialState[relative_index + node_index] = initialStateCopy[node_index]

        if save_history:
            gpu_stateHistory[
                (step + 1) * gpu_threadNum[0] + relative_index + node_index
            ] = initialStateCopy[node_index]
