from bang.core.PBN import PBN
import bang.graph.graph as graph
import numpy as np

def find_attractors(Block :list[graph.PBN_Node], child_attractors :list[tuple[list[list[np.uint256]], list[graph.PBN_Node]]] = []) -> list[list[np.uint256]]:
    return []

def divide_and_counquer(network : PBN):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks()
    blocks :list[tuple[list[graph.PBN_Node], list[int]]] = PBN_graph.blocks #blocks shoudl be without influencers
    attractors :list[list[list[np.uint256]]] = []
    for block, chilren in blocks:
        if len(chilren) == 0:
            attractors.append(find_attractors(block))
        else:
            attractors.append(find_attractors(block, [(attractors[i], blocks[i][0]) for i in chilren]))
    return attractors[len(attractors) - 1]