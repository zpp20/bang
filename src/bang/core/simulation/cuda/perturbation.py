from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit(device=True)
def perform_perturbation(
    gpu_npLength,
    gpu_npNode,
    gpu_perturbation_rate,
    states,
    idx,
    initialStateCopy,
):
    indexState = 0
    start = 0
    perturbation = False

    # here, we iterate over intervals of nodes that can be perturbed
    # since npNodee is array of non-perturbable nodes, we iterate over intervals of nodes
    # not containing them;
    for t in range(gpu_npLength[0]):
        # for every perturbable node in the interval
        for node_index in range(start, gpu_npNode[t]):
            if xoroshiro128p_uniform_float32(states, idx) < gpu_perturbation_rate[0]:
                perturbation = True
                indexState = node_index // 32
                initialStateCopy[indexState] ^= 1 << (node_index % 32)

        start = gpu_npNode[t] + 1

    return perturbation
