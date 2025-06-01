import typing

import numba.cuda as cu
import numpy as np

if typing.TYPE_CHECKING:
    from bang.core import PBN

from bang.core.attractors.monolithic.segmentation import segment_attractors


def monolithic_detect_attractor(pbn: "PBN", initial_states, states_repr="bool", stream=None):
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
    if cu.is_available() and stream is None:
        stream = cu.default_stream()

    sync_pbn = (
        pbn if pbn.update_type == "synchronous" else pbn.clone_with(update_type="synchronous")
    )

    sync_pbn.set_states(initial_states, reset_history=True, stream=stream)

    history = sync_pbn.last_state

    state_bytes = tuple(state.tobytes() for state in sync_pbn.last_state)
    n_unique_states = len({state_bytes})
    last_n_unique_states = 0

    while n_unique_states != last_n_unique_states:
        sync_pbn.simple_steps(1)
        last_n_unique_states = n_unique_states
        state_bytes = tuple(state.tobytes() for state in sync_pbn.last_state)
        n_unique_states = len(set(state_bytes))
        history = np.hstack((history, sync_pbn.last_state))

    state_bytes_set = list(set(state_bytes))
    ret_list = [np.frombuffer(state, dtype=np.int32) for state in state_bytes_set]

    return segment_attractors(np.array(ret_list), history, pbn.n_nodes)
