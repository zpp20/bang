import typing
from itertools import product

import bang.core.attractors.blocks.graph as graph

if typing.TYPE_CHECKING:
    from bang.core import PBN

from bang.core.attractors.blocks.block_attractors import find_block_attractors
from bang.core.attractors.blocks.crossing import cross_attractors_cpu


def get_all_nodes(blocks: list[tuple[list[int], list[int]]], i: int) -> list[int]:
    if len(blocks[i][1]) == 0:
        return blocks[i][0]
    else:
        result = blocks[i][0].copy()
        for j in blocks[i][1]:
            result += get_all_nodes(blocks, j)
            return sorted(result)
    return []


def divide_and_conquer(network: "PBN"):
    PBN_graph = graph.Graph_PBN(network)
    PBN_graph.find_scc_and_blocks(dag_scc=True)
    blocks: list[tuple[list[int], list[int]]] = PBN_graph.blocks
    attractors: list[tuple[list[list[list[bool]]], list[int]]] = []
    max: int = 0
    for block, chilren in blocks:
        if len(chilren) == 0:
            attractors.append((find_block_attractors(network, block), block))
        else:
            res = set(block)
            for i in chilren:
                res.update(attractors[i][1])
            attractors.append(
                (
                    find_block_attractors(
                        network, block, [(attractors[i][0], attractors[i][1]) for i in chilren]
                    ),
                    list(res),
                )
            )

    for i in range(len(blocks)):
        if not any([i in children for block, children in blocks]):
            max = i
            break
    result, nodes = attractors[max]
    for i in range(max + 1, len(attractors)):
        result = [
            cross_attractors_cpu(x, nodes, y, attractors[i][1])[0]
            for x, y in product(result, attractors[i][0])
        ]
        nodes = list(set(nodes + attractors[i][1]))

    return result
