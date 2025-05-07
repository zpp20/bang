import random

import numpy as np
import numba

from bang.core.cuda.simulation import MAX_STATE_SIZE


@numba.jit()
def update_node(
    node_index,
    index_shift,
    index_state,
    rand,
    cum_cij,
    cum_nf,
    cum_nv,
    F,
    extra_f_index,
    extra_f,
    cum_extra_f,
    var_f,
    pow_num,
    initial_state_copy,
    initial_state,
):
    relative_index = 0

    # Choose function to update state
    while rand > cum_cij[cum_nf[node_index] + relative_index]:
        relative_index += 1

    start = cum_nf[node_index] + relative_index
    element_f = F[start]

    # Process variables in function
    start_var_f_index = cum_nv[start]
    result_state_size = cum_nv[start + 1] - start_var_f_index
    shift_num = 0

    for ind in range(result_state_size):
        relative_index = var_f[start_var_f_index + ind] // 32
        state_fragment = initial_state_copy[relative_index]
        if ((state_fragment >> (var_f[start_var_f_index + ind] % 32)) & 1) != 0:
            shift_num += pow_num[1][ind]

    # Handle extra functions if needed
    if shift_num > 32:
        tt = 0
        while extra_f_index[tt] != start:
            tt += 1
        element_f = extra_f[cum_extra_f[tt] + ((shift_num - 32) // 32)]
        shift_num = shift_num % 32

    element_f = element_f >> shift_num

    initial_state[index_state] ^= (-(element_f & 1) ^ initial_state_copy[index_state]) & (
        1 << (node_index - index_state * 32)
    )


@numba.jit()
def perform_perturbation(
    np_length,
    np_node,
    perturbation_rate,
    initial_state_copy,
):
    start = 0
    perturbation = False

    for t in range(np_length[0]):
        for node_index in range(start, np_node[t]):
            if random.random() < perturbation_rate[0]:
                perturbation = True
                index_state = node_index // 32
                initial_state_copy[index_state] ^= 1 << (node_index % 32)
        start = np_node[t] + 1

    return perturbation


@numba.jit()
def update_initial_state(
    thread_num,
    state_history,
    initial_state,
    state_size,
    idx,
    step,
    current_state,
    initial_state_copy,
    save_history,
):
    relative_index = state_size * idx

    initial_state_copy[:] = current_state[:]
    initial_state[relative_index : relative_index + state_size] = initial_state_copy[:]

    if save_history:
        history_start_index = (step + 1) * thread_num + relative_index
        state_history[history_start_index : history_start_index + state_size] = initial_state_copy[
            :
        ]


@numba.jit()
def cpu_converge_sync(
    state_history,
    thread_num,
    pow_num,
    cum_nf,
    cum_cij,
    node_num,
    perturbation_rate,
    cum_nv,
    F,
    var_f,
    initial_state,
    steps,
    state_size,
    extra_f,
    extra_f_index,
    cum_extra_f,
    np_length,
    np_node,
    save_history,
):
    state_size = state_size[0]
    thread_num = thread_num[0]
    steps = steps[0]

    for idx in range(thread_num):
        initial_state_copy = np.zeros(state_size, dtype=np.uint32)
        current_state = np.zeros(state_size, dtype=np.uint32)
        relative_index = idx * state_size

        # Initialize state
        initial_state_copy[:] = initial_state[relative_index : relative_index + state_size]
        current_state[:] = initial_state[relative_index : relative_index + state_size]

        if save_history:
            state_history[relative_index : relative_index + state_size] = current_state[:]

        for step in range(steps):
            perturbation = perform_perturbation(
                np_length,
                np_node,
                perturbation_rate,
                initial_state_copy,
            )

            if not perturbation:
                for node_index in range(node_num):
                    index_shift = 0
                    index_state = node_index // 32
                    rand = random.random()

                    update_node(
                        node_index,
                        index_shift,
                        index_state,
                        rand,
                        cum_cij,
                        cum_nf,
                        cum_nv,
                        F,
                        extra_f_index,
                        extra_f,
                        cum_extra_f,
                        var_f,
                        pow_num,
                        initial_state_copy,
                        current_state,
                    )

            update_initial_state(
                thread_num,
                state_history,
                initial_state,
                state_size,
                idx,
                step,
                current_state,
                initial_state_copy,
                save_history,
            )


@numba.jit()
def cpu_converge_async_one_random(
    state_history,
    thread_num,
    pow_num,
    cum_nf,
    cum_cij,
    node_num,
    perturbation_rate,
    cum_nv,
    F,
    var_f,
    initial_state,
    steps,
    state_size,
    extra_f,
    extra_f_index,
    cum_extra_f,
    np_length,
    np_node,
    save_history,
):
    state_size = state_size[0]
    thread_num = thread_num[0]
    steps = steps[0]

    for idx in range(thread_num):
        current_state = np.zeros(state_size, dtype=np.uint32)
        relative_index = idx * state_size

        # Initialize state
        current_state[:] = initial_state[relative_index : relative_index + state_size]

        if save_history:
            state_history[relative_index : relative_index + state_size] = current_state[:]

        for step in range(steps):
            perturbation = perform_perturbation(
                np_length,
                np_node,
                perturbation_rate,
                current_state,
            )

            if not perturbation:
                node_index = random.randint(0, node_num - 1)
                index_shift = node_index % 32
                index_state = node_index // 32
                rand = random.random()

                update_node(
                    node_index,
                    index_shift,
                    index_state,
                    rand,
                    cum_cij,
                    cum_nf,
                    cum_nv,
                    F,
                    extra_f_index,
                    extra_f,
                    cum_extra_f,
                    var_f,
                    pow_num,
                    current_state,
                    current_state,
                )

            update_initial_state(
                thread_num,
                state_history,
                initial_state,
                state_size,
                idx,
                step,
                current_state,
                current_state,
                save_history,
            )


@numba.jit()
def cpu_converge_async_random_order(
    state_history,
    thread_num,
    pow_num,
    cum_nf,
    cum_cij,
    node_num,
    perturbation_rate,
    cum_nv,
    F,
    var_f,
    initial_state,
    steps,
    state_size,
    extra_f,
    extra_f_index,
    cum_extra_f,
    np_length,
    np_node,
    save_history,
):
    state_size = state_size[0]
    thread_num = thread_num[0]
    steps = steps[0]

    for idx in range(thread_num):
        current_state = np.zeros(state_size, dtype=np.uint32)
        update_order = np.zeros(shape=(MAX_STATE_SIZE * 32,), dtype=np.uint32)
        relative_index = idx * state_size

        # Initialize state
        current_state[:] = initial_state[relative_index : relative_index + state_size]

        if save_history:
            state_history[relative_index : relative_index + state_size] = current_state[:]

        for step in range(steps):
            perturbation = perform_perturbation(
                np_length,
                np_node,
                perturbation_rate,
                current_state,
            )

            if not perturbation:
                for i in range(node_num):
                    update_order[i] = i

                for i in range(node_num - 1, 0, -1):
                    rand = random.randrange(0, 1)
                    j = int(rand * (i + 1))

                    update_order[i], update_order[j] = update_order[j], update_order[i]

                for i in range(node_num):
                    node_index = update_order[i]

                    index_shift = node_index % 32
                    index_state = node_index // 32

                    rand = random.randint(0, 1)

                    update_node(
                        node_index,
                        index_shift,
                        index_state,
                        rand,
                        cum_cij,
                        cum_nf,
                        cum_nv,
                        F,
                        extra_f_index,
                        extra_f,
                        cum_extra_f,
                        var_f,
                        pow_num,
                        current_state,
                        current_state,
                    )

            update_initial_state(
                thread_num,
                state_history,
                initial_state,
                state_size,
                idx,
                step,
                current_state,
                current_state,
                save_history,
            )
