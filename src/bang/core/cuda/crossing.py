import numpy as np
import numpy.typing as npt
from numba import cuda

def corss_attractors_gpu(
    attractor_list :list[npt.NDArray[np.int32]], 
    nodes :list[list[int]], 
    attractor_cum_index :list[npt.NDArray[np.int32]], 
    blocks_indices :list[int]
    ):
    attractors_global = [cuda.to_device(attractor) for attractor in attractor_list]
    attractors_len = cuda.to_device([attractor.shape[0] for attractor in attractor_list])
    attractors_index = cuda.to_device(np.array(attractor_cum_index))
    
    pass

def increment_states(current_states, max_vals) -> bool:
    current_states[0][0] += 1
    cf :int
    for i in range(len(max_vals) - 1):
        cf = current_states[i][0] / max_vals
        if cf != 0 and i == len(max_vals) - 1:
            break
        current_states[i][0] %= max_vals[i]
        current_states[i + 1][0] += cf
    return current_states[len(max_vals) - 1][0] == max_vals[len(max_vals) - 1]


# Only to be called with the number of threads equal to the number of combinations of attractors to be crossed
cuda.jit
def cross_attractors(
    attractors_global,
    attractors_cum_index,
    block_sizes, 
    select_order,
    result,
    result_start
    ):
    """
    Crosses a set of n attractors
    - attractors_global - a list global memory arrays each storing a list of all states in attractors in each block
    - `attractors_cum_index` - a cummulative index going into the second coordinate of `attractors_global`,  stores indices of each attractor
    - `block_sizes` - number of nodes in each relevant block
    - `select_order` - order by which nodes are to be selected from blocks
    - `result` - output array
    - `result_start` - where each thread should start writing 
    """
    comb_id :int = 1024 * cuda.blockIdx.x + cuda.threadIdx.x # type: ignore 
    tmp = comb_id
    atttractors = np.zeros(block_sizes.size) # cum_index of each attractor we are crossing
    lengths = atttractors.copy() # length of each attractor
    current_states = np.zeros((block_sizes.size, 3)) # for each block: (which node are we crossing, which integer to cross next, which bit to cross next)
    write_index :int = result_start
    
    for i in range(block_sizes.size):
        atttractors[i] = attractors_cum_index[(tmp % block_sizes[i])]
        lengths[i] = attractors_cum_index[(tmp % block_sizes[i]) + 1] - attractors_cum_index[(tmp % block_sizes[i])]
        tmp /= block_sizes[i]
        
    while True:
        new_node = np.zeros(select_order.size)
        to_read = 0
        for i in range(len(select_order)):
            # reads the bit pointed to by current states and stores it as 1 or 2
            read = 1 & (attractors_global[select_order[i]][current_states[select_order[i]][0]][current_states[select_order[i]][1]]) >> current_states[select_order[i]][2]
            new_node[i] += read << to_read
            current_states[select_order[i]][1] += (current_states[select_order[i]][1] % 32 + 1) / 32
            current_states[select_order[i]][2] = (current_states[select_order[i]][2] + 1) % 32
            to_read = (to_read + 1) % 32
        result[write_index][:] = new_node
        write_index += 1
        if not increment_states(current_states, lengths):
            break