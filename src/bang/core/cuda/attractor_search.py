from bang.core.PBN import PBN
import bang.graph.graph as graph
from itertools import product

def cross_states(state1 :list[bool], nodes1 :list[int], state2 :list[bool], nodes2 :list[int]) -> list[bool]:
    counter1, counter2 = 0, 0
    result :list[bool] = []
    for i in range(len(nodes1) + len(nodes2)):
        if nodes1[counter1] < nodes2[counter2]:
            result.append(state1[counter1])
            counter1 += 1
        else:
            result.append(state2[counter2])
            counter2 += 1
        if counter1 == len(nodes1):
            return result  + state2[counter2:]
        elif counter2 == len(nodes2):
            return result + state1[counter1:]
            
    return result

def cross_attractors(attractor1 :list[list[bool]], nodes1: list[int], 
                     attractor2 :list[list[bool]], nodes2: list[int]) -> tuple[list[list[bool]], list[int]]:
    result_nodes = nodes1 + nodes2
    result_nodes.sort()
            
    return [cross_states(x, nodes1, y,  nodes2) for x,y in product(attractor1, attractor2)], result_nodes

def find_attractors_realisation(network :PBN, initial_states :list[list[bool]], nodes :list[int]) -> list[list[list[bool]]]:
    def convert(state):
        node = 0
        result = []
        for i in range(network.n):
            if node < len(nodes) and nodes[node] == i:
                result.append(state[node])
                node += 1
            else:
                result.append(False)
        return result
    
    network.n_parallel = len(initial_states)            
    network.set_states([convert(state) for state in initial_states])
    
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

def find_block_attractors(network :PBN, Block :list[int], 
                                child_attractors :list[tuple[list[list[list[bool]]], list[int]]] = []) -> list[list[list[bool]]]:
    lengths :list[int] = [len(tup[0]) for tup in child_attractors]
    result = []
    
    if child_attractors == []:
        return find_attractors_realisation(network, states(Block), Block)
    
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
        result += find_attractors_realisation(network, *cross_attractors(states(Block), Block, attractor, nodes))
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