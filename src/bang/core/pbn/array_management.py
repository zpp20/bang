import numpy as np
import typing
from numba import cuda

from itertools import chain


if typing.TYPE_CHECKING:
    from bang.core import PBN


def convert_pbn_to_ndarrays(pbn: PBN, n_steps: int, save_history: bool = True):
    nf = pbn.getNf()
    nv = pbn.getNv()
    F = pbn.get_integer_f()
    varFInt = list(chain.from_iterable(pbn.getVarFInt()))
    cij = list(chain.from_iterable(pbn.getCij()))

    cumCij = np.cumsum(cij, dtype=np.float32)
    cumNv = np.cumsum([0] + nv, dtype=np.uint32)
    cumNf = np.cumsum([0] + nf, dtype=np.uint32)

    perturbation = pbn.getPerturbation()
    npNode = pbn.getNpNode()

    stateSize = pbn.stateSize()

    extraFCount = pbn.extraFCount()
    extraFIndexCount = pbn.extraFIndexCount()
    extraFIndex = pbn.extraFIndex()
    cumExtraF = pbn.cumExtraF()
    extraF = pbn.extraF()

    N = pbn.n_parallel

    initial_state = (
        np.zeros(N * stateSize, dtype=np.uint32)
        if pbn.latest_state is None
        else pbn.latest_state
    )
    initial_state = initial_state.reshape(N * stateSize)

    if save_history:
        state_history = np.zeros(N * stateSize * (n_steps + 1), dtype=np.uint32)
        state_history[: N * stateSize] = initial_state.copy()[:]

    else:
        state_history = np.zeros(0, dtype=np.uint32)

    cum_variable_count = np.array(cumNv, dtype=np.uint32)
    functions = np.array(F, dtype=np.uint32)
    function_variables = np.array(varFInt, dtype=np.uint32)
    thread_num = np.array([N], dtype=np.uint32)
    steps = np.array([n_steps], dtype=np.uint32)
    state_size = np.array([stateSize], dtype=np.uint32)
    extra_functions = np.array(extraF, dtype=np.uint32)
    extra_functions_index = np.array(extraFIndex, dtype=np.uint32)
    cum_extra_functions = np.array(cumExtraF, dtype=np.uint32)
    extra_function_count = np.array([extraFCount], dtype=np.uint32)
    extra_function_index_count = np.array([extraFIndexCount], dtype=np.uint32)
    perturbation_blacklist = np.array(npNode, dtype=np.uint32)
    non_perturbed_count = np.array([len(npNode)], dtype=np.uint32)
    function_probabilities = np.array(cumCij, dtype=np.float32)
    cum_function_count = np.array(cumNf, dtype=np.uint32)
    perturbation_rate = np.array([perturbation], dtype=np.float32)

    pow_num = np.zeros((2, 32), dtype=np.uint32)
    pow_num[1][0] = 1
    pow_num[0][0] = 0

    for i in range(1, 32):
        pow_num[0][i] = 0
        pow_num[1][i] = pow_num[1][i - 1] * 2

    return (
        state_history,
        thread_num,
        pow_num,
        cum_function_count,
        function_probabilities,
        perturbation_rate,
        cum_variable_count,
        functions,
        function_variables,
        initial_state,
        steps,
        state_size,
        extra_functions,
        extra_functions_index,
        cum_extra_functions,
        extra_function_count,
        extra_function_index_count,
        perturbation_blacklist,
        non_perturbed_count,
    )

class GpuMemoryContainer:

    def __init__(self, pbn: PBN, n_steps: int, save_history: bool = True):
        pbn_data = convert_pbn_to_ndarrays(pbn, n_steps, save_history)

        (
            state_history,
            thread_num,  # can vary between simple_steps executions
            pow_num,
            cum_function_count,
            function_probabilities,
            perturbation_rate,
            cum_variable_count,
            functions,
            function_variables,
            initial_state,  # usually varies between simple_steps executions
            steps,  # can vary between simple_steps executions
            state_size,
            extra_functions,
            extra_functions_index,
            cum_extra_functions,
            extra_function_count,
            extra_function_index_count,
            perturbation_blacklist,
            non_perturbed_count,
        ) = pbn_data

        self.gpu_cumNv = cuda.to_device(cum_variable_count)
        self.gpu_F = cuda.to_device(functions)
        self.gpu_varF = cuda.to_device(function_variables)
        self.gpu_initialState = cuda.to_device(initial_state)
        self.gpu_stateHistory = cuda.to_device(state_history)
        self.gpu_threadNum = cuda.to_device(thread_num)
        self.gpu_steps = cuda.to_device(steps)
        self.gpu_stateSize = cuda.to_device(state_size)
        self.gpu_extraF = cuda.to_device(extra_functions)
        self.gpu_extraFIndex = cuda.to_device(extra_functions_index)
        self.gpu_cumExtraF = cuda.to_device(cum_extra_functions)
        self.gpu_extraFCount = cuda.to_device(extra_function_count)
        self.gpu_extraFIndexCount = cuda.to_device(extra_function_index_count)
        self.gpu_npNode = cuda.to_device(perturbation_blacklist)
        self.gpu_npLength = cuda.to_device(non_perturbed_count)
        self.gpu_cumCij = cuda.to_device(function_probabilities)
        self.gpu_cumNf = cuda.to_device(cum_function_count)
        self.gpu_perturbation_rate = cuda.to_device(perturbation_rate)
        self.gpu_powNum = cuda.to_device(pow_num)
        