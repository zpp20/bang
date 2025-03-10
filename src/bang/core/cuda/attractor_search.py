from bang.core.PBN import PBN
import bang.graph.graph as graph
import numpy as np
from itertools import product

def cross_attractors(attractor1 :list[list[bool]], nodes1: list[int], 
                     attractor2 :list[list[bool]], nodes2: list[int]) -> tuple[list[list[bool]], list[int]]:
    result = nodes1 + nodes2
    result.sort()
    return [x + y for x,y in product(attractor1, attractor2)], result

def find_attractors_realisation(network :PBN, initial_states :list[list[bool]]) -> list[list[list[bool]]]:
    network.set_states(initial_states)
    
    n_unique_states = len(network.get_last_state())
    last_n_unique_states = 0
    
    while (n_unique_states != last_n_unique_states):
        network.simple_steps(1)
        last_n_unique_states = n_unique_states
        n_unique_states = len(network.get_last_state())
    return network.get_last_state()

def states(Block :list[int]) -> list[list[bool]]:
    states :list[list[bool]] = [[]]
    for node in Block:
        states = [state + [True] for state in states] + [state + [False] for state in states]
    return states

def cross(Block, attractor) -> list[list[bool]]:
    return cross_attractors(states(Block), Block, attractor, [])[0]

def find_block_attractors(network :PBN, Block :list[int], 
                                child_attractors :list[tuple[list[list[list[bool]]], list[int]]] = []) -> list[list[list[bool]]]:
    lengths :list[int] = [len(tup[0]) for tup in child_attractors]
    result = []
    
    if child_attractors == []:
        return find_attractors_realisation(network, states(Block))
    
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
    return result

def divide_and_counquer(network : PBN):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks()
    blocks :list[tuple[list[int], list[int]]] = PBN_graph.blocks #blocks shoudl be without influencers
    attractors :list[list[list[list[bool]]]] = []
    for block, chilren in blocks:
        if len(chilren) == 0:
            attractors.append(find_block_attractors(network, block))
        else:
            attractors.append(find_block_attractors(network, block, [(attractors[i], blocks[i][0]) for i in chilren]))
    return attractors[len(attractors) - 1]