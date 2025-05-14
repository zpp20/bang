import numba as nb

# Maximum state size of 16 means we can hold 32 * 16 = 512 nodes (size of uint32 * MAX_STATE_SIZE)
MAX_STATE_SIZE = 16
MAX_UPDATE_ORDER_SIZE = 512


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
