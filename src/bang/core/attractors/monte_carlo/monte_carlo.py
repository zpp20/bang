import random
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from bang.core import PBN

from .count_states import count_states
from .merge_attractors import merge_attractors

def monte_carlo(network : "PBN", trajectory_length : int, minimum_repetitions : int, initial_trajectory_length : int):
        """
        Detect attractors of a BN by monte carlo approach of running multiple trajectories while checking for repeat states in history.

        :param n_trajectories: Number of trajectories to be run simulataneously
        :type n_trajectories : int
        
        :param trajectories_len: Length of trajectories 
        :type trajectories_len: int


        """
        assert network._n_parallel < 2**network._n, "Warning! There are more concurrent trajectories than possible states"

        max_val = 2**network._n

        samples = random.sample(range(max_val), network._n_parallel)

        def int_to_bool_list(integer, int_size):
            return [(integer >> i) & 1 == 1 for i in range(int_size)]

        samples = [int_to_bool_list(sample, network._n) for sample in samples]

        network.set_states(states=samples, reset_history=True)
        
        network.save_history = False

        network.simple_steps(n_steps=initial_trajectory_length)

        network.save_history = True
        
        network.simple_steps(n_steps=trajectory_length)

        trajectories = network.history
        trajectories = np.squeeze(trajectories).T

        trajectories_state_count = count_states(trajectories)
        detected_attractors = []

        for i, trajectory in enumerate(trajectories_state_count):
            max_value = max(trajectory,key=trajectory.get)
            if trajectory[max_value] > minimum_repetitions:
                indices = np.where(trajectories[i] == max_value)[0]
                first_index = indices[0]
                attractor_trajectory = trajectories[i][first_index:]
                attractor_states = np.unique(attractor_trajectory)
                detected_attractors.append(attractor_states)
        
        attractors = merge_attractors(detected_attractors)

        return attractors