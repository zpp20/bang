import numpy as np


def segment_attractors(attractor_states, history, n_nodes):
    active_states = attractor_states
    transition = dict()

    for trajectory in history:
        for i in range(len(trajectory) - 1):
            if trajectory[i] in transition:
                if transition[trajectory[i]] != trajectory[i + 1]:
                    raise ValueError("Two different states from the same state")

            transition[trajectory[i]] = trajectory[i + 1]

    num_states = len(active_states)
    attractors = []

    while num_states > 0:
        initial_state = active_states[0][0] if n_nodes // 32 < 1 else active_states[0]

        rm_idx = np.where(active_states == initial_state)[0]
        active_states = np.delete(active_states, rm_idx)
        attractor_states = [initial_state]

        curr_state = transition[initial_state]
        while curr_state != initial_state:
            attractor_states.append(curr_state)

            rm_idx = np.where(active_states == curr_state)[0]
            active_states = np.delete(active_states, rm_idx)

            curr_state = transition[curr_state]

        attractors.append(attractor_states)
        num_states = len(active_states)

    return attractors
