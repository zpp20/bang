import numpy as np
import numpy.typing as npt
from numba import cuda
from functools import reduce
from operator import mul
     
def increment_states(current_states, max_vals) -> bool:
    current_states[0] += 1
    cf :int
    for i in range(len(max_vals) - 1):
        cf = current_states[i] / max_vals
        if cf != 0 and i == len(max_vals) - 1:
            break
        current_states[i] %= max_vals[i]
        current_states[i + 1] += cf
    return current_states[len(max_vals) - 1][0] == max_vals[len(max_vals) - 1]
     
def get_result(attractor_cum_index :list[npt.NDArray[np.int32]]):
    cum_result :list[int] = []
    result_size :int = 0
    max_lens :list[int] = []
    current = []
    for indices in attractor_cum_index:
        max_lens.append(indices.size)
        current.append(0)
    
    while True:
        tmp = reduce(mul, [attractor_cum_index[i][current[i]] - (attractor_cum_index[i][current[i] - 1] if current[i] > 0 else 0) for i in range(len(attractor_cum_index))])
        result_size += tmp
        cum_result.append(tmp)
        if not increment_states(current, max_lens):
            return cum_result[1:], result_size, reduce(mul, max_lens)
    

def corss_attractors_gpu(
    attractor_list :list[npt.NDArray[np.int32]], 
    attractor_cum_index :list[npt.NDArray[np.int32]], 
    nodes :list[list[int]]
    ):
    attractors_global = [cuda.to_device(attractor) for attractor in attractor_list]
    attractors_index = [cuda.to_device(attractor_cum_index[i]) for i in range(len(attractor_cum_index))]
    blocks_sizes = cuda.to_device(attractor.shape[0] for attractor in attractor_list)
    nodes_global = cuda.to_device(sorted(sum(nodes, [])))
    cum_result, result_size, threads = get_result(attractor_cum_index)
    result = np.zeros((result_size, 1))
    attractors = cuda.to_device(result)
    cross_attractors[1024, (threads // 1024) + 1]( # type: ignore
        attractors_global,
        attractor_cum_index,
        blocks_sizes,
        nodes_global,
        attractors,
        cuda.to_device(cum_result)
    )
    return attractors


# Only to be called with the number of threads equal to the number of combinations of attractors to be crossed
cuda.jit
def cross_attractors(
    attractors_global,
    attractors_cum_index,
    block_sizes, 
    nodes,
    result,
    result_start,
    bn_size
    ):
    """
    Crosses a set of n attractors
    - `attractors_global` - a list global memory arrays each storing a list of all states in attractors in each block
    - `attractors_cum_index` - a cummulative index going into the second coordinate of `attractors_global`,  stores indices of each attractor
    - `block_sizes` - number of states in attractors of each block
    - `select_order` - order by which nodes are to be selected from blocks
    - `result` - output array
    - `result_start` - where each thread should start writing 
    """
    comb_id :int = 1024 * cuda.blockIdx.x + cuda.threadIdx.x # type: ignore 
    tmp = comb_id
    atttractors = np.zeros(block_sizes.size) # cum_index of each attractor we are crossing
    lengths = atttractors.copy() # length of each attractor
    current_states = np.zeros((block_sizes.size, )) # for each block: (which node are we crossing, which integer to cross next, which bit to cross next)
    write_index :int = result_start
    
    for i in range(block_sizes.size):
        atttractors[i] = attractors_cum_index[i][(tmp % block_sizes[i])]
        lengths[i] = attractors_cum_index[i][(tmp % block_sizes[i]) + 1] - attractors_cum_index[i][(tmp % block_sizes[i])]
        tmp /= block_sizes[i]
        
    while True:
        new_node = np.zeros((bn_size // 32) + 1)
        to_read = 0
        for i in range(nodes.shape[0]):
            # reads the bit pointed to by current states and stores it as 1 or 2
            read = (1 << (nodes[i][1] % 32)) & (attractors_global[nodes[i][0]][current_states[nodes[i][0]]][nodes[i][1] // 32])
            new_node[nodes[i][1] // 32] += read
            to_read = (to_read + 1) % 32
        result[write_index][:] = new_node
        write_index += 1
        if not increment_states(current_states, lengths):
            break