from bang.core.PBN import PBN
import bang.graph.graph as graph
import numpy as np
from itertools import product

def cross_attractors(attractor1 :list[np.uint256], nodes1: list[graph.PBN_Node], 
                     attractor2 :list[np.uint256], nodes2: list[graph.PBN_Node]) -> tuple[list[np.uint256], list[graph.PBN_Node]]:
    return [x + y for x,y in product(attractor1, attractor2)], nodes1 + nodes2

def find_lower_sccs(network :PBN, initial_states :list[np.uint256]) -> list[list[np.uint256]]:
    return []

def states(Block :list[graph.PBN_Node]) -> list[np.uint256]:
    states = [np.uint256(0)]
    for node in Block:
        states = states + [state + np.power(2, node.id) for state in states]
    return states

def cross(Block, attractor) -> list[np.uint256]:
    return []

def find_realisation_attractors(network :PBN, Block :list[graph.PBN_Node], 
                                child_attractors :list[tuple[list[list[np.uint256]], list[graph.PBN_Node]]] = []) -> list[list[np.uint256]]:
    lengths :list[int] = [len(tup[0]) for tup in child_attractors]
    result = []
    
    indices = [0 for length in lengths]
    def inc():
        i = 0
        while True:
            if indices[i] < lengths[i]:
                indices[i] += 1
                return
            else:
                indices[i] = 0
    
    while indices != lengths:
        attractor, nodes = child_attractors[0][0][indices[0]], child_attractors[0][1]
        for i in range(len(child_attractors) - 1):
            attractor, nodes = cross_attractors(attractor, nodes, 
                             child_attractors[i + 1][0][indices[i + 1]], child_attractors[i + 1][1]) 
        result += find_lower_sccs(network, cross(Block, attractor))
        inc()
        pass
    return []

def divide_and_counquer(network : PBN):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks()
    blocks :list[tuple[list[graph.PBN_Node], list[int]]] = PBN_graph.blocks #blocks shoudl be without influencers
    attractors :list[list[list[np.uint256]]] = []
    for block, chilren in blocks:
        if len(chilren) == 0:
            attractors.append(find_realisation_attractors(network, block))
        else:
            attractors.append(find_realisation_attractors(network, block, [(attractors[i], blocks[i][0]) for i in chilren]))
    return attractors[len(attractors) - 1]