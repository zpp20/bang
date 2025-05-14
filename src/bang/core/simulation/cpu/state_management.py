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
