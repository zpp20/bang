import random

import numpy as np

from bang.core.simulation.common import update_node
from bang.core.simulation.cpu.perturbation import perform_perturbation
from bang.core.simulation.cpu.state_management import update_initial_state


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
    np.seterr(over="ignore")
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
