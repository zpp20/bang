from itertools import product

import bang.graph.graph as graph
from bang.core.PBN import PBN


def cross_states(
    state1: list[bool], nodes1: list[int], state2: list[bool], nodes2: list[int]
) -> list[bool]:
    counter1, counter2 = 0, 0
    result: list[bool] = []
    for i in range(len(nodes1) + len(nodes2)):
        if nodes1[counter1] < nodes2[counter2]:
            result.append(state1[counter1])
            counter1 += 1
        elif nodes1[counter1] > nodes2[counter2]:
            result.append(state2[counter2])
            counter2 += 1
        else:
            if state1[counter1] != state2[counter2]:
                return []
            result.append(state1[counter1])
            counter1 += 1
            counter2 += 1

        if counter1 == len(nodes1):
            return result + state2[counter2:]
        elif counter2 == len(nodes2):
            return result + state1[counter1:]

    return result


def cross_attractors(
    attractor1: list[list[bool]], nodes1: list[int], attractor2: list[list[bool]], nodes2: list[int]
) -> tuple[list[list[bool]], list[int]]:
    result_nodes = list(set(nodes1 + nodes2))
    result_nodes.sort()

    return [
        cross_states(x, nodes1, y, nodes2)
        for x, y in product(attractor1, attractor2)
        if len(cross_states(x, nodes1, y, nodes2)) != 0
    ], result_nodes


def apply(function, function_nodes, state: list[bool], nodes) -> bool:
    f_counter = len(function_nodes) - 1
    function_index: int = 0
    for i in range(len(nodes) - 1, -1, -1):
        if nodes[i] == function_nodes[f_counter]:
            function_index = (function_index << 1) + (1 if state[i] else 0)
            f_counter -= 1

        if f_counter == -1:
            break
    return function[function_index]


def find_attractors_realisation(
    network: PBN, initial_states: list[list[bool]], nodes: list[int]
) -> list[list[list[bool]]]:
    reduced_pbn = network.select_nodes(nodes)
    reduced_pbn.n_parallel = len(initial_states)
    attractor, history = reduced_pbn.detect_attractor(initial_states)
    attractors = reduced_pbn.segment_attractor(attractor, history)

    len_nodes = len(nodes)
    list_bools = [
        [[bool(num_state & 1 << i) for i in range(len_nodes)] for num_state in attractor_system]
        for attractor_system in attractors
    ]

    states = initial_states
    current_len, prev_len = len(initial_states), 0
    while current_len != prev_len:
        states = [
            [
                apply(network.F[nodes[i]], network.varFInt[nodes[i]], state, nodes)
                for i in range(len(nodes))
            ]
            for state in states
        ]
        unique_states = []
        for state in states:
            if state not in unique_states:
                unique_states.append(state)
        states = unique_states
        prev_len = current_len
        current_len = len(states)

    return list_bools


def states(Block: list[int]) -> list[list[bool]]:
    states: list[list[bool]] = [[]]
    for node in Block:
        states = [state + [True] for state in states] + [state + [False] for state in states]
    return states


def find_block_attractors(
    network: PBN,
    Block: list[int],
    child_attractors: list[tuple[list[list[list[bool]]], list[int]]] = [],
) -> list[list[list[bool]]]:
    lengths: list[int] = [len(tup[0]) for tup in child_attractors]
    result = []
    if child_attractors == []:
        return find_attractors_realisation(network, states(Block), Block)
    indices = [0 for length in lengths]

    def inc():
        i = 0
        while True:
            if indices[i] < lengths[i] - 1:
                indices[i] += 1
                return True
            else:
                indices[i] = 0
                i += 1
                if i == len(lengths):
                    return False

    while True:
        attractor, nodes = child_attractors[0][0][indices[0]], child_attractors[0][1]
        for i in range(len(child_attractors) - 1):
            attractor, nodes = cross_attractors(
                attractor,
                nodes,
                child_attractors[i + 1][0][indices[i + 1]],
                child_attractors[i + 1][1],
            )
        result += find_attractors_realisation(
            network, *cross_attractors(states(Block), Block, attractor, nodes)
        )
        if not inc():
            break
    return result


def get_all_nodes(blocks: list[tuple[list[int], list[int]]], i: int) -> list[int]:
    if len(blocks[i][1]) == 0:
        return blocks[i][0]
    else:
        result = blocks[i][0].copy()
        for j in blocks[i][1]:
            result += get_all_nodes(blocks, j)
            return sorted(result)
    return []


def divide_and_counquer(network: PBN):
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
            cross_attractors(x, nodes, y, attractors[i][1])[0]
            for x, y in product(result, attractors[i][0])
        ]
        nodes = list(set(nodes + attractors[i][1]))
    return result
