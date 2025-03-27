from bang.core.PBN import PBN
import bang.graph.graph as graph
from itertools import product
import numpy as np
import threading

def cross_attractors_gpu():
    pass

def calculate_realisations_gpu(starting):
    pass

def divide_and_counquer_gpu(network : PBN):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks()
    blocks :list[tuple[list[int], list[int]]] = PBN_graph.blocks
    semaphores = [threading.Semaphore(len(children)) for block, children in blocks]
    attractors :list[tuple[list[list[list[bool]]], list[int]]] = []
    max :int = 0
    result = None
    return result