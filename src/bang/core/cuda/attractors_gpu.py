from bang.core.PBN import PBN
import bang.graph.graph as graph
from itertools import product
import numpy as np
import numpy.typing as npt
import threading
from bang.core.cuda.crossing import corss_attractors_gpu
from functools import reduce
from operator import add

def states(Block: list[int], state_size) -> np.ndarray:
    states = np.zeros((2 ** len(Block), state_size))
    bool_states = [[]]
    for node in Block:
        bool_states = [state + [True] for state in bool_states] + [state + [False] for state in bool_states]
    for i in range(len(bool_states)):
        for j in Block:
            states[i][j // 32] &= states[i][j // 32] | (bool_states[i][j] << (j % 32))
    return states

def block_thread(
    id :int,
    pbn : PBN,
    sempaphore :threading.Semaphore, 
    parent_semaphores :list[threading.Semaphore], 
    blocks :list[tuple[list[int], list[int]]] ,
    attractors :list[npt.NDArray[np.int32]],
    attractors_cum_index :list[npt.NDArray[np.int32]],
    elementary_blocks :list[list[int]]
    ):
    initial_states = []
    for i in range(len(blocks[id][0])):
        sempaphore.acquire()
    if len(blocks[id][0]) != 0:
        initial_states = corss_attractors_gpu(
            [attractors[i] for i in blocks[id][0]],
            [attractors_cum_index[i] for i in blocks[id][0]],
            [elementary_blocks[i] for i in blocks[id][0]]
            )
    else:
        initial_states = states(elementary_blocks[id], pbn.n)
    attractors[id] = pbn.detect_attractor(initial_states)
    for sem in parent_semaphores:
        sem.release()
    pass

def get_elementary_blocks(blocks :list[tuple[list[int], list[int]]]) -> list[list[int]]:
    result :list[list[int]] = [] 
    for i in range(len(blocks)):
        result.append(blocks[i][0] + reduce(add, [result[j] for j in blocks[i][1]]))
    return result

def divide_and_counquer_gpu(network : PBN):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks(True)
    blocks :list[tuple[list[int], list[int]]] = PBN_graph.blocks
    elementery_blocks = get_elementary_blocks(blocks)
    semaphores = [threading.Semaphore(len(children)) for block, children in blocks]
    attractors :list[npt.NDArray[np.int32]] = [np.zeros((2, 2), dtype=np.int32) for block in blocks]
    attractors_cum_index :list[npt.NDArray[np.int32]] = [np.zeros((2, 2), dtype=np.int32) for block in blocks]
    threads :list[threading.Thread] = []
    
    for i in range(len(blocks)):
        children, block = blocks[i]
        threads.append(threading.Thread(target=block_thread, args=(
            i,
            network,
            semaphores[i], 
            [semaphores[j] for j in children], 
            blocks,
            attractors,
            attractors_cum_index,
            elementery_blocks
            )))
    map(lambda t : t.start(), threads)
    map(lambda t : t.join(), threads)
        
    return attractors[network.n - 1], attractors_cum_index[network.n - 1]