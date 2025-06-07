import numpy as np


def segment_attractors(attractor_states, history, n_nodes):
    byte_attractor_states = [state.tobytes().rstrip(b"\x00") for state in attractor_states]
    active_states = np.copy(byte_attractor_states)
    transition = dict()

    for trajectory in history:
        for i in range(len(trajectory) - 1):
            hashed_state = trajectory[i].tobytes().rstrip(b"\x00")
            next_hashed_state = trajectory[i + 1].tobytes().rstrip(b"\x00")
            if hashed_state in transition:
                if transition[hashed_state] != next_hashed_state:
                    raise ValueError("Two different states from the same state ")

            transition[hashed_state] = next_hashed_state
    num_states = len(active_states)
    attractors = []

    while num_states > 0:
        initial_state = active_states[0]
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

    int_attractors = [
        [np.frombuffer(state.ljust(4, b"\x00"), dtype=np.int32) for state in states]
        for states in attractors
    ]
    return int_attractors
