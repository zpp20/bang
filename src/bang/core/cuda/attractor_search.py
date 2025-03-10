from bang.core.PBN import PBN
import bang.graph.graph as graph
import numpy as np
from itertools import product

def cross_attractors(attractor1 :list[np.uint256], nodes1: list[int], 
                     attractor2 :list[np.uint256], nodes2: list[int]) -> tuple[list[np.uint256], list[int]]:
    result = nodes1 + nodes2
    result.sort()
    return [x + y for x,y in product(attractor1, attractor2)], result

def to_bool(integer :np.uint256, n) -> list[bool]:
    result :list[bool] = []
    for i in range(n):
        result.append(bool(integer % 2))
        integer = integer // 2
    return result
        

def find_attractors_realisation(network :PBN, initial_states :list[np.uint256]) -> list[list[np.uint256]]:
    states = [to_bool(state, network.getN()) for state in initial_states]
    network.set_states(states)
    
    n_unique_states = len(network.get_last_state())
    last_n_unique_states = 0
    
    while (n_unique_states != last_n_unique_states):
        network.simple_steps(1)
        last_n_unique_states = n_unique_states
        n_unique_states = len(network.get_last_state())
    return network.get_last_state()

def states(Block :list[int]) -> list[np.uint256]:
    states = [np.uint256(0)]
    for node in Block:
        states = states + [state + np.power(2, node) for state in states]
    return states

def cross(Block, attractor) -> list[np.uint256]:
    return cross_attractors(states(Block), Block, attractor, [])[0]

def find_realisation_attractors(network :PBN, Block :list[int], 
                                child_attractors :list[tuple[list[list[np.uint256]], list[int]]] = []) -> list[list[np.uint256]]:
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
        result += find_attractors_realisation(network, cross(Block, attractor))
        inc()
        pass
    return []

def divide_and_counquer(network : PBN):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks()
    blocks :list[tuple[list[int], list[int]]] = PBN_graph.blocks #blocks shoudl be without influencers
    attractors :list[list[list[np.uint256]]] = []
    for block, chilren in blocks:
        if len(chilren) == 0:
            attractors.append(find_realisation_attractors(network, block))
        else:
            attractors.append(find_realisation_attractors(network, block, [(attractors[i], blocks[i][0]) for i in chilren]))
    return attractors[len(attractors) - 1]