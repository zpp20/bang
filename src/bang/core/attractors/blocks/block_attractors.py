import typing

from bang.core.attractors.blocks.crossing import cross_attractors_cpu
from bang.core.attractors.blocks.node_selection import select_nodes

if typing.TYPE_CHECKING:
    from bang.core import PBN


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
    network: "PBN", initial_states: list[list[bool]], nodes: list[int]
) -> list[list[list[bool]]]:
    reduced_pbn = select_nodes(network, nodes)
    reduced_pbn._n_parallel = len(initial_states)

    attractors = reduced_pbn.monolithic_detect_attractors(initial_states, repr="int")

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
                apply(network._f[nodes[i]], network._var_f_int[nodes[i]], state, nodes)
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
    network: "PBN",
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
            attractor, nodes = cross_attractors_cpu(
                attractor,
                nodes,
                child_attractors[i + 1][0][indices[i + 1]],
                child_attractors[i + 1][1],
            )
        result += find_attractors_realisation(
            network, *cross_attractors_cpu(states(Block), Block, attractor, nodes)
        )
        if not inc():
            break
    return result
