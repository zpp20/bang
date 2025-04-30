from itertools import product

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

