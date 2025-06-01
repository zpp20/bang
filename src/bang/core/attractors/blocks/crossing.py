from functools import reduce
from itertools import accumulate, product
from operator import add, mul

import numpy as np
import numpy.typing as npt
from numba import cuda


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


def cross_attractors_cpu(
    attractor1: list[list[bool]], nodes1: list[int], attractor2: list[list[bool]], nodes2: list[int]
) -> tuple[list[list[bool]], list[int]]:
    result_nodes = list(set(nodes1 + nodes2))
    result_nodes.sort()

    return [
        cross_states(x, nodes1, y, nodes2)
        for x, y in product(attractor1, attractor2)
        if len(cross_states(x, nodes1, y, nodes2)) != 0
    ], result_nodes


def increment_states(current_states, max_vals) -> bool:
    current_states[0] += 1
    cf: int
    for i in range(len(max_vals) - 1):
        cf = current_states[i] // max_vals[i]
        if cf == 0 and i == len(max_vals) - 1:
            break
        current_states[i] %= max_vals[i]
        current_states[i + 1] += cf
    return current_states[-1] != max_vals[-1]


# TODO: figure out vectorize
@cuda.jit(device=True)
def increment_states2(current_states, max_vals) -> bool:
    current_states[0] += 1
    cf: int
    for i in range(len(max_vals) - 1):
        cf = current_states[i] // max_vals[i]
        if cf == 0 and i == len(max_vals) - 1:
            break
        current_states[i] %= max_vals[i]
        current_states[i + 1] += cf
    return current_states[-1] != max_vals[-1]


def get_result(attractor_cum_index: list[npt.NDArray[np.uint32]]):
    cum_result: list[int] = []
    result_size: int = 0
    max_lens: list[int] = []
    current = []
    for indices in attractor_cum_index:
        max_lens.append(indices.size - 1)
        current.append(0)
    while True:
        tmp = reduce(
            mul,
            [
                attractor_cum_index[i][current[i] + 1] - attractor_cum_index[i][current[i]]
                for i in range(len(attractor_cum_index))
            ],
        )
        result_size += tmp
        cum_result.append(result_size)
        if not increment_states(current, max_lens):
            return (
                [np.uint32(0)] + cum_result,
                result_size,
                list(accumulate(max_lens, add)),
                reduce(mul, max_lens),
            )


def cross_attractors_gpu(
    attractor_list: list[npt.NDArray[np.uint32]],
    attractor_cum_index: list[npt.NDArray[np.uint32]],
    nodes: list[list[tuple[int, int]]],
    stream=None,
    int_size=1,
):
    if stream is None:
        stream = cuda.default_stream()

    attractors_global = cuda.to_device(
        np.concatenate(attractor_list, dtype=np.uint32), stream=stream
    )
    attractors_index = cuda.to_device(
        np.array(
            list(
                accumulate(
                    attractor_cum_index,
                    lambda x, y: np.concatenate(
                        (x, np.array(list(map(lambda z: z + x[-1], y[1:])), dtype=np.uint32))
                    ),
                )
            )[-1],
            dtype=np.uint32,
        ),
        stream=stream,
    )
    blocks_sizes = cuda.to_device(
        np.array([attractor.shape[0] for attractor in attractor_list], dtype=np.uint32),
        stream=stream,
    )
    nodes_global = cuda.to_device(np.array(sorted(sum(nodes, [])), dtype=np.uint32), stream=stream)
    cum_result, result_size, block_attractor_size, threads = get_result(attractor_cum_index)
    result = np.zeros((result_size, int_size), dtype=np.uint32)
    attractors = cuda.to_device(result, stream=stream)
    indices = cuda.to_device(np.empty((3, 1024 * ((threads // 1024) + 1), blocks_sizes.size), dtype=np.uint32))  # type: ignore
    cross_attractors[1024, (threads // 1024) + 1, stream](  # type: ignore
        attractors_global,
        attractors_index,
        blocks_sizes,
        nodes_global,
        attractors,
        cuda.to_device(np.array(cum_result, dtype=np.uint32), stream=stream),
        indices,
        threads,
        cuda.to_device(np.array([0] + block_attractor_size, dtype=np.uint32), stream=stream),
    )
    return attractors.copy_to_host(), cum_result


# Only to be called with the number of threads equal to the number of combinations of attractors to be crossed
@cuda.jit
def cross_attractors(
    attractors_global,
    attractors_cum_index,
    block_sizes: np.ndarray,
    nodes,
    result,
    result_start,
    indices,
    threads,
    max_lens,
):
    """
    Crosses a set of n attractors
    - `attractors_global` - a list global memory arrays each storing a list of all states in attractors in each block
    - `attractors_cum_index` - a cummulative index going into the second coordinate of `attractors_global`,  stores indices of each attractor
    - `block_sizes` - number of states in attractors of each block
    - `select_order` - order by which nodes are to be selected from blocks
    - `result` - output array
    - `result_start` - where each thread should start writing
    """
    comb_id: int = 1024 * cuda.blockIdx.x + cuda.threadIdx.x  # type: ignore
    tmp = comb_id
    write_index: int = result_start[comb_id]

    if comb_id >= threads:
        return

    for i in range(block_sizes.size):
        indices[0][comb_id][i] = (
            attractors_cum_index[max_lens[i] + (tmp % block_sizes[i]) + 1]
            - attractors_cum_index[max_lens[i] + (tmp % block_sizes[i])]
        )
        indices[1][comb_id][i] = attractors_cum_index[max_lens[i] + (tmp % block_sizes[i])]
        indices[2][comb_id][i] = 0
        tmp //= block_sizes[i]

    while True:
        to_read = 0
        for i in range(nodes.shape[0]):
            # reads the bit pointed to by current states and stores it as 1 or 2
            read = (1 << (nodes[i][0] % 32)) & (
                attractors_global[
                    indices[1][comb_id][nodes[i][1]] + indices[2][comb_id][nodes[i][1]]
                ][nodes[i][0] // 32]
            )
            result[write_index][nodes[i][0] // 32] += read
            to_read = (to_read + 1) % 32
        write_index += 1
        if not increment_states2(indices[2][comb_id], indices[0][comb_id]):
            break
