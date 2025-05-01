import numpy as np

from bang.core.attractors.monolithic.segmentation import segment_attractors

from bang.core import PBN


def monolithic_detect_attractor(pbn: "PBN", initial_states):
    """
    Detects all atractor states in PBN

    Parameters
    ----------

    initial_states : list[list[Bool]]
        List of investigated states.
    Returns
    -------
    attractors : list[list[bool]]
        List of states where attractors are coded as ints

    """

    pbn.set_states(initial_states, reset_history=True)
    history = pbn.get_last_state()

    state_bytes = tuple(state.tobytes() for state in pbn.get_last_state())
    n_unique_states = len({state_bytes})
    last_n_unique_states = 0

    while n_unique_states != last_n_unique_states:
        pbn.simple_steps(1)
        last_n_unique_states = n_unique_states
        # print("Last state: ", self.get_last_state())
        state_bytes = tuple(state.tobytes() for state in pbn.get_last_state())
        n_unique_states = len(set(state_bytes))
        history = np.hstack((history, pbn.get_last_state()))

    state_bytes_set = list(set(state_bytes))
    ret_list = [np.frombuffer(state, dtype=np.uint32)[0] for state in state_bytes_set]

    return segment_attractors(np.array(ret_list), history)
