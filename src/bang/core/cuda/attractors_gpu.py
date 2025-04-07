from bang.core.PBN import PBN
import bang.graph.graph as graph
from itertools import product
import numpy as np
import numpy.typing as npt
import threading
from bang.core.cuda.crossing import corss_attractors_gpu

def find_attractors_gpu():
    pass

def block_thread(
    id :int,
    sempaphore :threading.Semaphore, 
    parent_semaphores :list[threading.Semaphore], 
    children :list[int],
    attractors :list[tuple[npt.NDArray[np.int32], list[int]]],
    attractors_cum_index :list[npt.NDArray[np.int32]]
    ):
    
    for i in range(len(children)):
        sempaphore.acquire()
    if len(children) != 0:
        corss_attractors_gpu(
            [attractors[i][0] for i in children],
            [],
            [],
            []
            )
    find_attractors_gpu()
    for sem in parent_semaphores:
        sem.release()
    pass

def get_elementary_blocks(blocks :list[tuple[list[int], list[int]]]) -> list[list[int]]:
    pass

def divide_and_counquer_gpu(network : PBN):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks()
    blocks :list[tuple[list[int], list[int]]] = PBN_graph.blocks
    starting_blocks = get_elementary_blocks(blocks)
    semaphores = [threading.Semaphore(len(children)) for block, children in blocks]
    attractors :list[tuple[npt.NDArray[np.int32], list[int]]] = []
    attractors_cum_index :list[npt.NDArray[np.int32]]
    threads :list[threading.Thread] = []
    
    for i in range(len(blocks)):
        children, block = blocks[i]
        threads.append(threading.Thread(target=block_thread, args=(semaphores[i], [semaphores[j] for j in children], i in starting_blocks)))
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
    pass