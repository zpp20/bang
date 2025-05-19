import random


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
