import random
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from bang.core import PBN

def _merge_attractors(network : "PBN", attractors):
        sets = [set(arr) for arr in attractors]
        merged = []

        while sets:
            current, *rest = sets
            current = set(current)

            changed = True
            while changed:
                changed = False
                remaining = []
                for s in rest:
                    if current & s:  # If there is any overlap
                        current |= s
                        changed = True
                    else:
                        remaining.append(s)
                rest = remaining

            merged.append(np.array(list(current)))
            sets = rest

        return merged

def monte_carlo(network : "PBN", num_steps : int, step_length : int, repetitions : int):
        """
        Detect attractors of a BN by monte carlo approach of running multiple trajectories while checking for repeat states in history.

        :param n_trajectories: Number of trajectories to be run simulataneously
        :type n_trajectories : int
        
        :param trajectories_len: Length of trajectories 
        :type trajectories_len: int


        """
        assert network.n_parallel < 2**network.n, "Warning! There are more concurrent trajectories than possible states"

        max_val = 2**network.n

        samples = random.sample(range(max_val), network.n_parallel)

        def int_to_bool_list(integer, int_size):
            return [(integer >> i) & 1 == 1 for i in range(int_size)]

        samples = [int_to_bool_list(sample, network.n) for sample in samples]

        network.set_states(states=samples, reset_history=True)

        for s in range(num_steps):
            network.simple_steps(n_steps=step_length)

        trajectories = network.get_trajectories()
        trajectories = np.squeeze(trajectories).T

        trajectories_state_count = network._count_states(trajectories)
        detected_attractors = []

        for i, trajectory in enumerate(trajectories_state_count):
            max_value = max(trajectory,key=trajectory.get)
            if trajectory[max_value] > repetitions:
                indices = np.where(trajectories[i] == max_value)[0]
                first_index = indices[0]
                attractor_trajectory = trajectories[i][first_index:]
                attractor_states = np.unique(attractor_trajectory)
                detected_attractors.append(attractor_states)
        
        attractors = _merge_attractors(network, detected_attractors)

        return attractors