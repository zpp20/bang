from bang.core.PBN import PBN
import bang.graph.graph as graph
from itertools import product
import numpy as np
import threading

def cross_attractors_gpu():
    pass

def calculate_realisations_gpu():
    pass

def block_thread(sempaphore : threading.Semaphore, parent_semaphores :list[threading.Semaphore], starting: bool):
    pass

def get_elementary_blocks(blocks :list[tuple[list[int], list[int]]]) -> list[int]:
    pass

def divide_and_counquer_gpu(network : PBN):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks()
    blocks :list[tuple[list[int], list[int]]] = PBN_graph.blocks
    starting_blocks = get_elementary_blocks(blocks)
    semaphores = [threading.Semaphore(len(children)) for block, children in blocks]
    attractors :list[tuple[list[list[list[bool]]], list[int]]] = []
    threads :list[threading.Thread] = []
    
    for i in range(len(blocks)):
        children, block = blocks[i]
        threads.append(threading.Thread(target=block_thread, args=(semaphores[i], [semaphores[j] for j in children], i in starting_blocks)))
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
    pass