from bang.core.PBN import PBN
import bang.graph.graph as graph
from itertools import product
import numpy as np
import numpy.typing as npt
import threading
from bang.core.cuda.crossing import corss_attractors_gpu
from functools import reduce
from operator import add
import numba.cuda

def states(Block: list[int], state_size) -> np.ndarray:
    states = np.zeros((2 ** len(Block), (state_size // 32) + 1), dtype=np.int32)
    bool_states = [[]]
    for node in Block:
        bool_states = [state + [False] for state in bool_states] + [state + [True] for state in bool_states]
    for i in range(len(bool_states)):
        for j in range(len(Block)):
            states[i][Block[j] // 32] |= (bool_states[i][j] << (Block[j] % 32))
    return states

def block_thread(
    id :int,
    pbn : PBN,
    semaphores :list[threading.Semaphore], 
    blocks :list[tuple[list[int], list[int]]] ,
    attractors :list[npt.NDArray[np.int32]],
    attractors_cum_index :list[npt.NDArray[np.int32]],
    elementary_blocks :list[list[int]]
    ):
    
    initial_states = []
    thread_stream = numba.cuda.stream()
    
    for i in range(len(blocks[id][1])):
        semaphores[id].acquire()
        
    if len(blocks[id][1]) != 0:
        initial_states = corss_attractors_gpu(
            [attractors[i] for i in blocks[id][1]] + [states(blocks[id][0], pbn.n)],
            [attractors_cum_index[i] for i in blocks[id][1]] + [np.array([0, 2 ** len(blocks[id][0])], dtype=np.int32)],
            [list(zip(elementary_blocks[i], [i] * len(elementary_blocks[i]))) for i in blocks[id][1]] + [list(zip(blocks[id][0], len(blocks[id][0]) * [id]))]
            )[0]
    else:
        initial_states = states(elementary_blocks[id], pbn.n)
    
    attractors_tmp = [pbn.detect_attractor(initial_states, True, thread_stream)[0]]# pbn.segment_attractor(*pbn.detect_attractor(initial_states, True, thread_stream))

    attractors_cum_index[id] = np.zeros((len(attractors_tmp) + 1,), dtype=np.int32)    
    for i in range(len(attractors_tmp)):
        attractors_cum_index[id][i] = attractors_cum_index[id][i - 1] + len(attractors_tmp[i]) if i > 0 else 0
    
    attractors[id] = np.zeros((attractors_cum_index[id][-2] + len(attractors_tmp[-1]), (pbn.n // 32) + 1), dtype=np.int32)
    attractors_cum_index[id][-1] = attractors[id].shape[0]
    
    for i in range(len(attractors_tmp)):
        attractors[id][attractors_cum_index[id][i]:attractors_cum_index[id][i + 1]] = attractors_tmp[i]
    
    for sem in [semaphores[j] for j in range(len(blocks)) if id in blocks[j][1]]:
        sem.release()
    pass

def get_elementary_blocks(blocks :list[tuple[list[int], list[int]]]) -> list[list[int]]:
    result :list[list[int]] = [] 
    for i in range(len(blocks)):
        result.append(blocks[i][0] + reduce(add, [[]] + [result[j] for j in blocks[i][1]]))
    return result

def divide_and_counquer_gpu(network : PBN):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks(True)
    blocks :list[tuple[list[int], list[int]]] = PBN_graph.blocks
    elementery_blocks = get_elementary_blocks(blocks)
    semaphores = [threading.Semaphore(0) for block, children in blocks]
    attractors :list[npt.NDArray[np.int32]] = [np.zeros((2, 2), dtype=np.int32) for block in blocks]
    attractors_cum_index :list[npt.NDArray[np.int32]] = [np.zeros((2, 2), dtype=np.int32) for block in blocks]
    threads :list[threading.Thread] = []
    
    for i in range(len(blocks)):
        threads.append(threading.Thread(target=block_thread, args=(
            i,
            network,
            semaphores, 
            blocks,
            attractors,
            attractors_cum_index,
            elementery_blocks
            )))
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
        
    return attractors[len(blocks) - 1], attractors_cum_index[len(blocks) - 1]