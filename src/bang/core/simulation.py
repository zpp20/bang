import time
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numba as nb
from itertools import chain
from . import PBN
from . import device_info as info


@cuda.jit
def kernel_converge(gpu_stateHistory, gpu_threadNum, gpu_powNum, gpu_cumNf, gpu_cumCij, states, nodeNum, gpu_perturbation_rate, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_mean, gpu_steps, gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF, gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode):
    idx = cuda.grid(1)

    steps = gpu_steps[0]
    stateSize = gpu_stateSize[0]

    initialStateCopy = cuda.local.array(shape=(10,), dtype=nb.int32)
    initialState = cuda.local.array(shape=(4,), dtype=nb.int32)

    relative_index = idx * stateSize

    for node_index in range(stateSize):
        initialStateCopy[node_index] = gpu_initialState[relative_index + node_index]
        initialState[node_index] = initialStateCopy[node_index]

    for node_index in range(stateSize):
        initialStateCopy[node_index] = initialState[node_index]

        relative_index = stateSize * idx
        gpu_stateHistory[relative_index + node_index] = initialState[node_index]

    steps = gpu_steps[0]

    for step in range(steps):
        perturbation = False
        indexShift = 0
        indexState = 0
        start = 0

        for t in range(gpu_npLength[0]):
            for node_index in range(start, gpu_npNode[t]):
                rand = xoroshiro128p_uniform_float32(states, idx)

                if rand < gpu_perturbation_rate[0]:
                    perturbation = True
                    indexState = node_index // 32
                    indexShift = indexState * 32
                initialStateCopy[indexState] = initialStateCopy[indexState] ^ (1 << (node_index - indexShift))

            start = gpu_npNode[t] + 1

        if not perturbation:
            for node_index in range(nodeNum):
                if (indexShift == 32):
                    indexState += 1
                    indexShift = 0

                rand = xoroshiro128p_uniform_float32(states, idx)
                relative_index = 0

                while rand > gpu_cumCij[gpu_cumNf[node_index] + relative_index]:
                    relative_index += 1

                start = gpu_cumNf[node_index] + relative_index

                element_f = gpu_F[start]
                start_var_f_index = gpu_cumNv[start]
                result_state_size = gpu_cumNv[start + 1] - start_var_f_index
                shift_num = 0

                for ind in range(result_state_size):
                    relative_index = gpu_varF[start_var_f_index + ind] // 32
                    state_fragment = initialStateCopy[relative_index]

                    if ((state_fragment >> (gpu_varF[start_var_f_index + ind]) % 32) & 1) != 0:
                        shift_num += gpu_powNum[1][ind]

                if shift_num > 32:
                    tt = 0

                    while gpu_extraFIndex[tt] != start:
                        tt += 1

                    element_f = gpu_extraF[gpu_cumExtraF[tt] + ((shift_num - 32) // 32)]
                    shift_num = shift_num % 32

                element_f = element_f >> shift_num

                initialState[indexState] ^= (-(element_f & 1) ^ initialStateCopy[indexState]) & (1 << (node_index - indexState * 32))
                indexShift += 1

                for node_index in range(stateSize):
                    initialStateCopy[node_index] = initialState[node_index]

                relative_index = stateSize * idx
                gpu_stateHistory[(step + 1) * gpu_threadNum[0] + relative_index + node_index] = initialState[node_index]

    relative_index = stateSize * idx

    for node_index in range(stateSize):
        gpu_initialState[relative_index + node_index] = initialState[node_index]


def german_gpu_run(pbn: PBN, steps, trajectories):
    n = pbn.getN()
    nf = pbn.getNf()
    nv = pbn.getNv()
    F = list(chain.from_iterable(pbn.getF()))
    varFInt = list(chain.from_iterable(pbn.getVarFInt()))
    cij = list(chain.from_iterable(pbn.getCij()))

    cumCij = np.cumsum(cij, dtype=np.float32)
    cumNv = np.cumsum([0] + nv, dtype=np.int32)
    cumNf = np.cumsum([0] + nf, dtype=np.int32)

    perturbation = pbn.getPerturbation()
    npNode = pbn.getNpNode()

    stateSize = pbn.stateSize()
    extraFInfo = pbn.getFInfo()

    extraFCount = extraFInfo[0]
    extraFIndexCount = extraFInfo[1]
    extraFIndex = extraFInfo[2]
    cumExtraF = extraFInfo[3]
    extraF = extraFInfo[4]

    PBN_memory_size = 0
    PBN_memory_size += len(cumNf) * np.uint16().itemsize
    
    if (cumNf[n] + 1) % 2 != 0:
        PBN_memory_size += np.uint16().itemsize

    PBN_memory_size += len(F) * np.int32().itemsize
    PBN_memory_size += len(varFInt) * np.uint16().itemsize

    if len(varFInt) != 0:
        PBN_memory_size += np.uint16().itemsize

    if extraFCount != 0:
        PBN_memory_size += len(extraFIndex) * np.uint32().itemsize
        PBN_memory_size += len(extraF) * np.uint32().itemsize
        PBN_memory_size += len(cumExtraF) * np.uint32().itemsize

    PBN_memory_size += (len(npNode) + 1) *  np.uint32().itemsize

    block, blockSize = info._compute_device_info(PBN_memory_size, stateSize, steps, trajectories)
    print("Number of blocks", block, "  Block size: ", blockSize)
    N = block * blockSize
    gpu_cumNv = cuda.to_device(np.array(cumNv, dtype=np.uint16))
    gpu_F = cuda.to_device(np.array(F, dtype=np.int32))
    gpu_varF = cuda.to_device(np.array(varFInt, dtype=np.uint16))
    gpu_initialState = cuda.to_device(np.zeros(N * stateSize, dtype=np.int32))
    gpu_stateHistory = cuda.to_device(np.zeros(N * stateSize * (steps + 1), dtype=np.int32))
    gpu_threadNum = cuda.to_device(np.array([N], dtype=np.uint16))
    gpu_steps = cuda.to_device(np.array([steps], dtype=np.int32))
    gpu_stateSize = cuda.to_device(np.array([stateSize], dtype=np.int32))
    gpu_extraF = cuda.to_device(np.array(extraF, dtype=np.int32))
    gpu_extraFIndex = cuda.to_device(np.array(extraFIndex, dtype=np.uint16))
    gpu_cumExtraF = cuda.to_device(np.array(cumExtraF, dtype=np.uint16))
    gpu_extraFCount = cuda.to_device(np.array([extraFCount], dtype=np.int32))
    gpu_extraFIndexCount = cuda.to_device(np.array([extraFIndexCount], dtype=np.int32))
    gpu_npNode = cuda.to_device(np.array(npNode, dtype=np.int32))
    gpu_npLength = cuda.to_device(np.array([len(npNode)], dtype=np.int32))
    gpu_cumCij = cuda.to_device(np.array(cumCij, dtype=np.float32))
    gpu_cumNf = cuda.to_device(np.array(cumNf, dtype=np.int32))
    gpu_perturbation_rate = cuda.to_device(np.array([perturbation], dtype=np.float32))

    pow_num = np.zeros((2, 32), dtype=np.int32)
    pow_num[1][0] = 1
    pow_num[0][0] = 0

    for i in range(1, 32):
        pow_num[0][i] = 0
        pow_num[1][i] = pow_num[1][i - 1] * 2

    gpu_powNum = cuda.to_device(pow_num)

    states = create_xoroshiro128p_states(N, seed=time.time())

    kernel_converge[block, blockSize](gpu_stateHistory, gpu_threadNum, gpu_powNum, gpu_cumNf, gpu_cumCij, states, n, gpu_perturbation_rate, gpu_cumNv, gpu_F, gpu_varF, gpu_initialState, gpu_steps, gpu_stateSize, gpu_extraF, gpu_extraFIndex, gpu_cumExtraF, gpu_extraFCount, gpu_extraFIndexCount, gpu_npLength, gpu_npNode)

    last_state = gpu_initialState.copy_to_host()
    history = gpu_stateHistory.copy_to_host()

    print(last_state)
    print(last_state.shape)
    print(history.reshape((steps + 1, -1)))



