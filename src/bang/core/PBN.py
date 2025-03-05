from typing import List
import os
from math import floor
import numpy as np
import numpy.typing as npt
import math
from itertools import chain
from bang.core.cuda.simulation import kernel_converge
import numba
from numba import cuda

from bang.parsing.assa import load_assa
from bang.parsing.sbml import parseSBMLDocument

from numba.cuda.random import create_xoroshiro128p_states
import datetime


class PBN:

    def __init__(
        self,
        n: int,
        nf: List[int],
        nv: List[int],
        F: List[List[bool]],
        varFInt: List[List[int]],
        cij: List[List[float]],
        perturbation: float,
        npNode: List[int],
        n_parallel: int = 512,
    ):
        self.n = n  # the number of nodes
        self.nf = nf  # the size is n
        self.nv = nv  # the sizef = cum(nf)
        self.F = F  # each element of F stores a column of the truth table "F"， e.g., F.get(0)=[true false], the length of the element is 2^nv(boolean function index)
        self.varFInt = varFInt
        self.cij = cij  # the size=n, each element represents the selection probability of a node, and therefore the size of each element equals to the number of functions of each node
        self.perturbation = perturbation  # perturbation rate
        self.npNode = npNode  # index of those nodes without perturbation. To make simulation easy, the last element of npNode will always be n, which also indicate the end of the list. If there is only one element, it means there is no disabled node.
        # TODO, informacje z GPU powinny posłużyć żeby ustawić sensowny default
        self.n_parallel = n_parallel

        self.history: np.ndarray | None = None
        self.latest_state: np.ndarray | None = None

    def __str__(self):
        return f"PBN(n={self.n}, nf={self.nf}, nv={self.nv}, F={self.F}, varFInt={self.varFInt}, cij={self.cij}, perturbation={self.perturbation}, npNode={self.npNode})"

    def getN(self):
        return self.n

    def getNf(self):
        return self.nf

    def getNv(self):
        return self.nv

    def _get_integer_functions(self, f: list[bool], extra_functions: list[int]) -> int:
        """
        based on fromVector function from original ASSA-PBN
        converts list of bools to 32 bit int with bits representing truth table
        extra bits are placed into extraF list
        """

        retval = 0
        i = 0
        prefix = 0
        tempLen = len(f)
        if tempLen > 32:  # we have to add values to extraF
            for i in range(32):
                if f[i + prefix]:
                    retval |= 1 << i

            prefix += 32
            tempLen -= 32

        else:  # we just return proper into
            for i in range(tempLen):
                if f[i]:
                    retval |= 1 << i

            return retval

        while (
            tempLen > 0
        ):  # switched condition to tempLen > 0 to get one more iteration after tempLen > 32 is false
            other = 0
            for i in range(32):
                if f[i + prefix]:
                    other |= 1 << i

            prefix += 32
            tempLen -= 32
            extra_functions.append(other)

        return retval

    def get_last_state(self) -> np.ndarray | None:
        return self.latest_state

    def get_trajectories(self) -> np.ndarray | None:
        return self.history

    @staticmethod
    def _bools_to_state_array(bools: list[bool], node_count: int) -> np.ndarray:
        state_size = PBN._calc_state_size(node_count)
        """
        Converts list of bools to integer
        """
        if len(bools) != node_count:
            raise ValueError("Number of bools must be equal to number of nodes")

        integer_state = np.zeros((state_size), dtype=np.int32)

        for i in range(state_size)[::-1]:
            for bit in range((len(bools) - 32 * i) % 32):
                if bools[i * 32 + bit]:
                    integer_state[i] |= 1 << bit

        return integer_state

    def set_states(self, states: list[list[bool]], reset_history=True):
        """
        Sets the initial states of the PBN.

        Parameters
        ----------
        states : list[list[bool]]
            List of states to be set.
        reset_history : bool, optional
            If True, the history of the PBN will be reset. Defaults to True.
        """
        converted_states = [
            self._bools_to_state_array(state, self.n)
            for state in states
        ]

        self.latest_state = np.array(converted_states)

        if reset_history:
            self.history = np.array([converted_states])

    def extraFCount(self):
        """
        Returns number of extraFs
        """
        extraFCount = 0
        for elem in self.nv:
            if elem > 5:
                extraFCount += 2 ** (elem - 5) - 1

        return extraFCount

    def extraFIndexCount(self):
        """
        Returns number of extraFIndex
        """
        extraFIndexCount = 0

        for elem in self.nv:
            if elem > 5:
                extraFIndexCount += 1

        return extraFIndexCount

    def extraFIndex(self):
        """
        Returns list of extraFIndex
        """
        extraFIndex = []

        for i in range(len(self.nv)):
            if self.nv[i] > 5:
                extraFIndex.append(i)

        return extraFIndex

    def cumExtraF(self):
        """
        Returns list of cumExtraF
        """
        cumExtraF = [0]

        for i in range(len(self.nv)):
            if self.nv[i] > 5:
                cumExtraF.append(cumExtraF[-1] + 2 ** (self.nv[i] - 5) - 1)

        return cumExtraF

    def extraF(self):
        """
        Returns list of extraFs
        """
        extraF = []

        for i in range(len(self.F)):
            extraF.append(self._get_integer_functions(self.F[i], extraF))

        return extraF

    def cumulativeNumberFunctions(self) -> np.ndarray:
        """
        Returns sum of all elements in nf
        """
        return np.cumsum([0] + self.nf)

    def cumulativeNumberVariables(self) -> np.ndarray:
        """
        Returns sum of all elements in nv
        """

        return np.cumsum([0] + self.nv)

    @staticmethod
    def _calc_state_size(node_count: int) -> int:
        return math.ceil(node_count / 32)

    def stateSize(self) -> int:
        """
        Returns number of 32 bit integers needed to store all variables
        """
        return self._calc_state_size(self.n)

    def getF(self) -> list[list[bool]]:
        return self.F

    def get_integer_f(self):
        return [self._get_integer_functions(func, []) for func in self.F]

    def getVarFInt(self):
        return self.varFInt

    def getCij(self):
        return self.cij

    def getPerturbation(self):
        return self.perturbation

    def getNpNode(self):
        return self.npNode
    
    def reduce_F(self, states : List[List[int]]):
        """
        Reduces truth tables of PBN by removing states that does not change

        Parameters
        ----------

        states : List[int]
            List of investigated states. States are lists of int with length n
            where i-th index represents i-th variable. 0 represents False and 1 represents True.
        Returns
        -------
        active_variables : List[List[int]]
            List of indeces of variables that change between states

        F_reduced : List[List[Bool]]
            Truth tables with removed constant variables
        """
        initial_state = states[0]

        constant_vars = {i for i in range(0, self.n)}
        
        for state in states[1:]: 
            for var in range(0, self.n):
                if initial_state[var] != state[var]:
                    constant_vars.remove(var)
        
        # print("constant - ", constant_vars)
        new_F = list()
        new_varF = list()
        for F_func, F_vars in zip(self.F, self.varFInt):
            #assumes F contains truthtables for sorted vars
            # print("F_vars ", F_vars)
            new_varF.append(list())
            new_F.append(list())
            curr_num_vars = len(F_vars)
            curr_F = F_func
            curr_vars = F_vars
            current_removed = 0
            for i, var in enumerate(F_vars):
                if var in constant_vars:
                    curr_i = i - current_removed
                    var_state = initial_state[var]
                    # indeces = [j + (2**curr_i) * (j // (2**curr_i)) + (curr_i + 1) * var_state for j in range(2**(curr_num_vars - 1))]
                    # print("indeces - ", indeces, " var_state ", var_state, " curr_i ", curr_i)
                    curr_F = [curr_F[j + (2**curr_i) * (j // (2**curr_i)) + (curr_i + 1) * var_state] for j in range(2**(curr_num_vars - 1))]
                    curr_num_vars -= 1
                    current_removed += 1
                else:
                    new_varF[-1].append(var)

            new_F[-1].append(curr_F)

        return new_varF, new_F

    @staticmethod
    def _perturb_state_by_actions(actions: npt.NDArray[np.uint32], state: np.ndarray | None) -> np.ndarray:
        if state is None:
            raise ValueError("State must be set before explicit perturbation")

        copystate = state.copy()

        for index in actions:
            state_index = index // 32
            bit_index = index % 32

            if state_index >= state.size:
                raise IndexError("State index out of bounds")

            copystate[:, state_index] ^= 1 << bit_index
            
        return copystate

    def simple_steps(self, n_steps, actions: npt.NDArray[np.uint] | None = None):
        if self.latest_state is None or self.history is None:
            raise ValueError("Initial state must be set before simulation")
        
        if actions is not None:
            self.latest_state = self._perturb_state_by_actions(actions, self.latest_state)
            self.history = np.concatenate([self.history, self.latest_state], axis=1)

        n = self.getN()
        nf = self.getNf()
        nv = self.getNv()
        F = self.get_integer_f()
        varFInt = list(chain.from_iterable(self.getVarFInt()))
        cij = list(chain.from_iterable(self.getCij()))

        cumCij = np.cumsum(cij, dtype=np.float32)
        cumNv = np.cumsum([0] + nv, dtype=np.int32)
        cumNf = np.cumsum([0] + nf, dtype=np.int32)

        perturbation = self.getPerturbation()
        npNode = self.getNpNode()

        stateSize = self.stateSize()

        extraFCount = self.extraFCount()
        extraFIndexCount = self.extraFIndexCount()
        extraFIndex = self.extraFIndex()
        cumExtraF = self.cumExtraF()
        extraF = self.extraF()

        # TODO, tutaj powinno być wyciąganie informacji z GPU
        block = self.n_parallel // 32
        blockSize = 32

        N = self.n_parallel

        initial_state = np.zeros(N * stateSize, dtype=np.int32) if self.latest_state is None else self.latest_state
        initial_state = initial_state.reshape(N * stateSize)

        gpu_cumNv = cuda.to_device(np.array(cumNv, dtype=np.int32))
        gpu_F = cuda.to_device(np.array(F, dtype=np.int32))
        gpu_varF = cuda.to_device(np.array(varFInt, dtype=np.int32))
        gpu_initialState = cuda.to_device(np.zeros(N * stateSize, dtype=np.int32))
        gpu_stateHistory = cuda.to_device(np.zeros(N * stateSize * (n_steps + 1), dtype=np.int32))
        gpu_threadNum = cuda.to_device(np.array([N], dtype=np.int32))
        gpu_mean = cuda.to_device(np.zeros((N, 2), dtype=np.float32))
        gpu_steps = cuda.to_device(np.array([n_steps], dtype=np.int32))
        gpu_stateSize = cuda.to_device(np.array([stateSize], dtype=np.int32))
        gpu_extraF = cuda.to_device(np.array(extraF, dtype=np.int32))
        gpu_extraFIndex = cuda.to_device(np.array(extraFIndex, dtype=np.int32))
        gpu_cumExtraF = cuda.to_device(np.array(cumExtraF, dtype=np.int32))
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

        states = create_xoroshiro128p_states(N, seed=numba.uint64(datetime.datetime.now().timestamp()))

        kernel_converge[block, blockSize]( # type: ignore
            gpu_stateHistory,
            gpu_threadNum,
            gpu_powNum,
            gpu_cumNf,
            gpu_cumCij,
            states,
            n,
            gpu_perturbation_rate,
            gpu_cumNv,
            gpu_F,
            gpu_varF,
            gpu_initialState,
            gpu_mean,
            gpu_steps,
            gpu_stateSize,
            gpu_extraF,
            gpu_extraFIndex,
            gpu_cumExtraF,
            gpu_extraFCount,
            gpu_extraFIndexCount,
            gpu_npLength,
            gpu_npNode,
        )

        last_state = gpu_initialState.copy_to_host()
        run_history = gpu_stateHistory.copy_to_host()

        self.latest_state = last_state.reshape((N, stateSize))

        run_history = run_history.reshape((N, n_steps + 1, stateSize))

        if self.history is not None:
            self.history = np.concatenate([self.history, run_history[:, 1:, :]], axis=1)
        else:
            self.history = run_history


def load_sbml(path: str) -> tuple:
        return parseSBMLDocument(path)


def load_from_file(path, format="sbml"):
    """
    Loads a PBN from files of format .pbn or .sbml.

    Parameters
    ----------
    path : str
        Path to the file of format .pbn.
    format : str, optional
        Choose the format. Can be either 'sbml' for files with .sbml format
        or 'assa' for files with .pbn format. Defaults to 'sbml'.

    Returns
    -------
    PBN
        PBN object representing the network from the file.
    """
    match format:
        case "sbml":
            return PBN(*load_sbml(path))
        case "assa":
            return PBN(*load_assa(path))
        case _:
            raise ValueError("Invalid format")




