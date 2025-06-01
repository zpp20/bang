import random
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from bang.core import PBN

# from .count_states import count_states
from bang.core.attractors.monte_carlo.merge_attractors import merge_attractors


def monte_carlo(network: "PBN", initial_trajectory_length: int, trajectory_length: int):
    """
    Detect attractors of a BN by monte carlo approach of running multiple trajectories while checking for repeat states in history.

    :param n_trajectories: Number of trajectories to be run simulataneously
    :type n_trajectories : int

    :param trajectories_len: Length of trajectories
    :type trajectories_len: int


    """
    assert (
        network._n_parallel < 2**network._n
    ), "Warning! There are more concurrent trajectories than possible states"

    samples = [
        [random.choice([True, False]) for _ in range(network._n)]
        for _ in range(network._n_parallel)
    ]

    network.set_states(states=samples, reset_history=True)
    network.save_history = True

    network.simple_steps(n_steps=initial_trajectory_length)

    network.save_history = True

    network.simple_steps(n_steps=trajectory_length)

    trajectories = network.history
    trajectories = np.squeeze(trajectories).T
    trajectories = trajectories[::, 1:]

    # trajectories_state_count = count_states(trajectories)
    # detected_attractors = []

    # for i, trajectory in enumerate(trajectories_state_count):
    #     max_value = max(trajectory,key=trajectory.get)
    #     if trajectory[max_value] > minimum_repetitions:
    #         indices = np.where(trajectories[i] == max_value)[0]
    #         first_index = indices[0]
    #         attractor_trajectory = trajectories[i][first_index:]
    #         attractor_states = np.unique(attractor_trajectory)
    #         detected_attractors.append(attractor_states)

    attractors = merge_attractors(trajectories)

    return attractors
