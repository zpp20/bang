"""
Module containing the PBN class and helpers.
"""

import datetime
import math
from enum import Enum
from itertools import chain
from typing import List, Literal

import graphviz
import numba
import numpy as np
import numpy.typing as npt
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

import bang.graph
import bang.visualization
from bang.core.cuda.simulation import kernel_converge_async, kernel_converge_sync
from bang.parsing.assa import load_assa
from bang.parsing.sbml import parseSBMLDocument


class updateType(Enum):
    """
    Enum representing the type of update.
    """

    ASYNCHRONOUS = 0
    SYNCHRONOUS = 1


class PBN:
    """Class representing the PBN and the execution of simulations on it.

    :param n: The number of nodes.
    :type n: int
    :param nf: The size of each node.
    :type nf: List[int]
    :param nv: The size of each node's truth table.
    :type nv: List[int]
    :param F: The truth table of each node.
    :type F: List[List[bool]]
    :param varFInt: The index of each node's truth table.
    :type varFInt: List[List[int]]
    :param cij: The selection probability of each node.
    :type cij: List[List[float]]
    :param perturbation: The perturbation rate.
    :type perturbation: float
    :param npNode: Index of nodes without perturbation.
    :type npNode: List[int]
    :param n_parallel: The number of parallel simulations. Defaults to 512.
    :type n_parallel: int, optional
    :param history: The execution history of the PBN, tracking the states of all trajectories.
    :type history: np.ndarray
    :param latest_state: The last encountered state of the PBN's trajectories.
    :type latest_state: np.ndarray
    :param previous_simulations: List of previous simulations.
    :type previous_simulations: List[np.ndarray]
    """

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
        update_type_int: int = 1,  ## TODO change to 0, synchronous shouldnt be default
    ):
        self.n = n
        self.nf = nf
        self.nv = nv
        self.F = F
        self.varFInt = varFInt
        self.cij = cij
        self.perturbation = perturbation
        self.npNode = list(sorted(npNode))
        self.n_parallel = n_parallel
        self.update_type = updateType(update_type_int)
        self.history: np.ndarray = np.zeros((1, n_parallel, self.stateSize()), dtype=np.int32)
        self.latest_state: np.ndarray = np.zeros((n_parallel, self.stateSize()), dtype=np.int32)
        self.previous_simulations: List[np.ndarray] = []

    def __str__(self):
        return f"PBN(n={self.n}, nf={self.nf}, nv={self.nv}, F={self.F}, varFInt={self.varFInt}, cij={self.cij}, perturbation={self.perturbation}, npNode={self.npNode})"

    def getN(self) -> int:
        """
        Returns the number of nodes.

        :returns: The number of nodes.
        :rtype: int
        """
        return self.n

    def getNf(self) -> List[int]:
        """
        Returns the size of each node.

        :returns: The size of each node.
        :rtype: List[int]
        """
        return self.nf

    def getNv(self) -> List[int]:
        """
        Returns the size of each node's truth table.

        :returns: The size of each node's truth table.
        :rtype: List[int]
        """
        return self.nv

    def _get_integer_functions(self, f: List[bool], extra_functions: List[int]) -> int:
        """
        Converts list of bools to 32-bit int with bits representing truth table.

        :param f: List of boolean values representing the truth table.
        :type f: List[bool]
        :param extra_functions: List to store extra functions if the truth table exceeds 32 bits.
        :type extra_functions: List[int]
        :returns: Integer representation of the truth table.
        :rtype: int
        """
        retval = 0
        i = 0
        prefix = 0
        tempLen = len(f)
        if tempLen > 32:
            for i in range(32):
                if f[i + prefix]:
                    retval |= 1 << i

            prefix += 32
            tempLen -= 32

        else:
            for i in range(tempLen):
                if f[i]:
                    retval |= 1 << i

            return retval

        while tempLen > 0:
            other = 0
            for i in range(32):
                if f[i + prefix]:
                    other |= 1 << i

            prefix += 32
            tempLen -= 32
            extra_functions.append(other)

        return retval

    def get_last_state(self) -> np.ndarray:
        """
        Returns the last encountered state of the PBN's trajectories.

        :returns: The last encountered state of the PBN's trajectories.
        :rtype: np.ndarray
        """
        return self.latest_state

    def get_trajectories(self) -> np.ndarray:
        """
        Returns the execution history of the PBN, tracking the states of all trajectories.

        :returns: The execution history of the PBN.
        :rtype: np.ndarray
        """
        return self.history

    def save_trajectories(self, filename: str):
        """
        Saves the execution history of the PBN to a CSV file.

        :param filename: The name of the file to save the history.
        :type filename: str
        """
        np.save(filename, self.history)

    def save_last_state(self, filename: str):
        """
        Saves the last encountered state of the PBN's trajectories to a CSV file.

        :param filename: The name of the file to save the last state.
        :type filename: str
        """
        np.save(filename, self.latest_state)

    def get_blocks(self) -> list[list[int]]:
        """
        Returns the blocks of the PBN.

        :returns: The blocks of the PBN.
        :rtype: list[list[int]]
        """
        return bang.graph.get_blocks(self)

    @staticmethod
    def _bools_to_state_array(bools: List[bool], node_count: int) -> np.ndarray:
        """
        Converts list of bools to integer array.

        :param bools: List of boolean values representing the state.
        :type bools: List[bool]
        :param node_count: Number of nodes.
        :type node_count: int
        :raises ValueError: If the number of bools is not equal to the number of nodes.
        :returns: Integer array representing the state.
        :rtype: np.ndarray
        """
        state_size = PBN._calc_state_size(node_count)
        if len(bools) != node_count:
            raise ValueError("Number of bools must be equal to number of nodes")

        integer_state = np.zeros((state_size), dtype=np.int32)

        for i in range(state_size)[::-1]:
            for bit in range((len(bools) - 32 * i) % 32):
                if bools[i * 32 + bit]:
                    integer_state[i] |= 1 << bit

        return integer_state

    def set_states(self, states: List[List[bool]], reset_history: bool = False):
        """
        Sets the initial states of the PBN. If the number of trajectories is different than the number of previous trajectories,
        the history will be pushed into `self.previous_simulations` and the active history will be reset.

        :param states: List of states to be set.
        :type states: List[List[bool]]
        :param reset_history: If True, the history of the PBN will be reset. Defaults to False.
        :type reset_history: bool, optional
        """
        converted_states = [self._bools_to_state_array(state, self.n) for state in states]

        self.n_parallel = len(states)
        self.latest_state = np.array(converted_states).reshape((self.n_parallel, self.stateSize()))

        if reset_history:
            self.history = np.array(converted_states).reshape(
                (1, self.n_parallel, self.stateSize())
            )
        else:
            if len(states) != self.history.shape[1]:
                self.previous_simulations.append(self.history.copy())

                self.history = np.array(converted_states).reshape(
                    (1, self.n_parallel, self.stateSize())
                )
            else:
                self.history = np.concatenate(
                    [
                        self.history,
                        np.array(converted_states).reshape((1, self.n_parallel, self.stateSize())),
                    ],
                    axis=0,
                )

    def extraFCount(self) -> int:
        """
        Returns the number of extra functions.

        :returns: The number of extra functions.
        :rtype: int
        """
        extraFCount = 0
        for elem in self.nv:
            if elem > 5:
                extraFCount += 2 ** (elem - 5) - 1

        return extraFCount

    def extraFIndexCount(self) -> int:
        """
        Returns the number of extra function indices.

        :returns: The number of extra function indices.
        :rtype: int
        """
        extraFIndexCount = 0

        for elem in self.nv:
            if elem > 5:
                extraFIndexCount += 1

        return extraFIndexCount

    def extraFIndex(self) -> List[int]:
        """
        Returns a list of extra function indices.

        :returns: List of extra function indices.
        :rtype: List[int]
        """
        extraFIndex = []

        for i in range(len(self.nv)):
            if self.nv[i] > 5:
                extraFIndex.append(i)

        return extraFIndex

    def cumExtraF(self) -> List[int]:
        """
        Returns a list of cumulative extra functions.

        :returns: List of cumulative extra functions.
        :rtype: List[int]
        """
        cumExtraF = [0]

        for i in range(len(self.nv)):
            if self.nv[i] > 5:
                cumExtraF.append(cumExtraF[-1] + 2 ** (self.nv[i] - 5) - 1)

        return cumExtraF

    def extraF(self) -> List[int]:
        """
        Returns a list of extra functions.

        :returns: List of extra functions.
        :rtype: List[int]
        """
        extraF = []

        for i in range(len(self.F)):
            extraF.append(self._get_integer_functions(self.F[i], extraF))

        return extraF

    def cumulativeNumberFunctions(self) -> np.ndarray:
        """
        Returns the cumulative sum of all elements in nf.

        :returns: Cumulative sum of all elements in nf.
        :rtype: np.ndarray
        """
        return np.cumsum([0] + self.nf)

    def cumulativeNumberVariables(self) -> np.ndarray:
        """
        Returns the cumulative sum of all elements in nv.

        :returns: Cumulative sum of all elements in nv.
        :rtype: np.ndarray
        """
        return np.cumsum([0] + self.nv)

    @staticmethod
    def _calc_state_size(node_count: int) -> int:
        """
        Calculates the number of 32-bit integers needed to store all variables.

        :param node_count: Number of nodes.
        :type node_count: int
        :returns: Number of 32-bit integers needed to store all variables.
        :rtype: int
        """
        return math.ceil(node_count / 32)

    def stateSize(self) -> int:
        """
        Returns the number of 32-bit integers needed to store all variables.

        :returns: Number of 32-bit integers needed to store all variables.
        :rtype: int
        """
        return self._calc_state_size(self.n)

    def getF(self) -> List[List[bool]]:
        """
        Returns the truth table of each node.

        :returns: The truth table of each node.
        :rtype: List[List[bool]]
        """
        return self.F

    def get_integer_f(self) -> List[int]:
        """
        Returns the integer representation of the truth table for each node.

        :returns: Integer representation of the truth table for each node.
        :rtype: List[int]
        """
        return [self._get_integer_functions(func, []) for func in self.F]

    def getVarFInt(self) -> List[List[int]]:
        """
        Returns the index of each node's truth table.

        :returns: The index of each node's truth table.
        :rtype: List[List[int]]
        """
        return self.varFInt

    def getCij(self) -> List[List[float]]:
        """
        Returns the selection probability of each node.

        :returns: The selection probability of each node.
        :rtype: List[List[float]]
        """
        return self.cij

    def getPerturbation(self) -> float:
        """
        Returns the perturbation rate.

        :returns: The perturbation rate.
        :rtype: float
        """
        return self.perturbation

    def getNpNode(self) -> List[int]:
        """
        Returns the index of nodes without perturbation.

        :returns: The index of nodes without perturbation.
        :rtype: List[int]
        """
        return self.npNode

    def reduce_F(self, states: List[List[int]]) -> tuple:
        """
        Reduces truth tables of PBN by removing states that do not change.

        :param states: List of investigated states. States are lists of int with length n where i-th index represents i-th variable. 0 represents False and 1 represents True.
        :type states: List[List[int]]
        :returns: Tuple containing list of indices of variables that change between states and truth tables with removed constant variables.
        :rtype: tuple
        """
        initial_state = states[0]

        constant_vars = {i for i in range(0, self.n)}

        for state in states[1:]:
            for var in range(0, self.n):
                if initial_state[var] != state[var]:
                    constant_vars.remove(var)

        new_F = list()
        new_varF = list()
        for F_func, F_vars in zip(self.F, self.varFInt):
            new_varF.append(list())
            new_F.append(list())
            curr_num_vars = len(F_vars)
            curr_F = F_func

            # curr_vars = F_vars

            current_removed = 0
            for i, var in enumerate(F_vars):
                if var in constant_vars:
                    curr_i = i - current_removed
                    var_state = initial_state[var]
                    curr_F = [
                        curr_F[j + (2**curr_i) * (j // (2**curr_i)) + (curr_i + 1) * var_state]
                        for j in range(2 ** (curr_num_vars - 1))
                    ]
                    curr_num_vars -= 1
                    current_removed += 1
                else:
                    new_varF[-1].append(var)

            new_F[-1].append(curr_F)

        return new_varF, new_F

    def pbn_data_to_np_arrays(self, n_steps: int):
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

        N = self.n_parallel

        initial_state = (
            np.zeros(N * stateSize, dtype=np.int32)
            if self.latest_state is None
            else self.latest_state
        )
        initial_state = initial_state.reshape(N * stateSize)

        cum_variable_count = np.array(cumNv, dtype=np.int32)
        functions = np.array(F, dtype=np.int32)
        function_variables = np.array(varFInt, dtype=np.int32)
        state_history = np.zeros(N * stateSize * (n_steps + 1), dtype=np.int32)
        thread_num = np.array([N], dtype=np.int32)
        steps = np.array([n_steps], dtype=np.int32)
        state_size = np.array([stateSize], dtype=np.int32)
        extra_functions = np.array(extraF, dtype=np.int32)
        extra_functions_index = np.array(extraFIndex, dtype=np.int32)
        cum_extra_functions = np.array(cumExtraF, dtype=np.int32)
        extra_function_count = np.array([extraFCount], dtype=np.int32)
        extra_function_index_count = np.array([extraFIndexCount], dtype=np.int32)
        perturbation_blacklist = np.array(npNode, dtype=np.int32)
        non_perturbed_count = np.array([len(npNode)], dtype=np.int32)
        function_probabilities = np.array(cumCij, dtype=np.float32)
        cum_function_count = np.array(cumNf, dtype=np.int32)
        perturbation_rate = np.array([perturbation], dtype=np.float32)

        pow_num = np.zeros((2, 32), dtype=np.int32)
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

    @staticmethod
    def _perturb_state_by_actions(
        actions: npt.NDArray[np.uint32], state: np.ndarray | None
    ) -> np.ndarray:
        """
        Perturbs the state by performing the given actions.

        :param actions: Array of actions to be performed on the state.
        :type actions: npt.NDArray[np.uint32]
        :param state: The current state of the PBN.
        :type state: np.ndarray or None
        :raises ValueError: If the state is not set before explicit perturbation.
        :returns: The perturbed state.
        :rtype: np.ndarray
        """
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

    def simple_steps(self, n_steps: int, actions: npt.NDArray[np.uint] | None = None):
        """
        Simulates the PBN for a given number of steps.

        :param n_steps: Number of steps to simulate.
        :type n_steps: int
        :param actions: Array of actions to be performed on the PBN. Defaults to None.
        :type actions: npt.NDArray[np.uint], optional
        :raises ValueError: If the initial state is not set before simulation.
        """
        if self.latest_state is None or self.history is None:
            raise ValueError("Initial state must be set before simulation")

        if actions is not None:
            self.latest_state = self._perturb_state_by_actions(actions, self.latest_state)
            self.history = np.concatenate([self.history, self.latest_state], axis=0)

        # Convert PBN data to numpy arrays
        pbn_data = self.pbn_data_to_np_arrays(n_steps)

        (
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
        ) = pbn_data

        gpu_cumNv = cuda.to_device(cum_variable_count)
        gpu_F = cuda.to_device(functions)
        gpu_varF = cuda.to_device(function_variables)
        gpu_initialState = cuda.to_device(initial_state)
        gpu_stateHistory = cuda.to_device(state_history)
        gpu_threadNum = cuda.to_device(thread_num)
        gpu_steps = cuda.to_device(steps)
        gpu_stateSize = cuda.to_device(state_size)
        gpu_extraF = cuda.to_device(extra_functions)
        gpu_extraFIndex = cuda.to_device(extra_functions_index)
        gpu_cumExtraF = cuda.to_device(cum_extra_functions)
        gpu_extraFCount = cuda.to_device(extra_function_count)
        gpu_extraFIndexCount = cuda.to_device(extra_function_index_count)
        gpu_npNode = cuda.to_device(perturbation_blacklist)
        gpu_npLength = cuda.to_device(non_perturbed_count)
        gpu_cumCij = cuda.to_device(function_probabilities)
        gpu_cumNf = cuda.to_device(cum_function_count)
        gpu_perturbation_rate = cuda.to_device(perturbation_rate)
        gpu_powNum = cuda.to_device(pow_num)

        states = create_xoroshiro128p_states(
            self.n_parallel, seed=numba.uint64(datetime.datetime.now().timestamp())
        )

        block = self.n_parallel // 32
        if block == 0:
            block = 1
        blockSize = 32

        if self.update_type == updateType.ASYNCHRONOUS:
            kernel_converge_async[block, blockSize](  # type: ignore
                gpu_stateHistory,
                gpu_threadNum,
                gpu_powNum,
                gpu_cumNf,
                gpu_cumCij,
                states,
                self.getN(),
                gpu_perturbation_rate,
                gpu_cumNv,
                gpu_F,
                gpu_varF,
                gpu_initialState,
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
        elif self.update_type == updateType.SYNCHRONOUS:
            kernel_converge_sync[block, blockSize](  # type: ignore
                gpu_stateHistory,
                gpu_threadNum,
                gpu_powNum,
                gpu_cumNf,
                gpu_cumCij,
                states,
                self.getN(),
                gpu_perturbation_rate,
                gpu_cumNv,
                gpu_F,
                gpu_varF,
                gpu_initialState,
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

        self.latest_state = last_state.reshape((self.n_parallel, self.stateSize()))

        run_history = run_history.reshape((n_steps + 1, self.n_parallel, self.stateSize()))

        if self.history is not None:
            self.history = np.concatenate([self.history, run_history[1:, :, :]], axis=0)
        else:
            self.history = run_history

    def detect_attractor(self, initial_states):
        """
        Detects all atractor states in PBN

        Parameters
        ----------

        initial_states : List[List[Bool]]
            List of investigated states.
        Returns
        -------
        attractor_states : np.array(int)
            List of states where attractors are coded as ints

        """

        self.set_states(initial_states, reset_history=True)
        history = self.get_last_state()

        state_bytes = tuple(state.tobytes() for state in self.get_last_state())
        n_unique_states = len({state_bytes})
        last_n_unique_states = 0

        while n_unique_states != last_n_unique_states:
            self.simple_steps(1)
            last_n_unique_states = n_unique_states
            # print("Last state: ", self.get_last_state())
            state_bytes = tuple(state.tobytes() for state in self.get_last_state())
            n_unique_states = len(set(state_bytes))
            history = np.hstack((history, self.get_last_state()))

        state_bytes_set = list(set(state_bytes))
        ret_list = [np.frombuffer(state, dtype=np.int32)[0] for state in state_bytes_set]
        return (np.array(ret_list), history)

    def segment_attractor(self, attractor_states, history):
        active_states = attractor_states
        transition = dict()

        for trajectory in history:
            # print(trajectory)
            for i in range(len(trajectory) - 1):
                if trajectory[i] in transition:
                    if transition[trajectory[i]] != trajectory[i + 1]:
                        raise ValueError("Two different states from the same state")

                transition[trajectory[i]] = trajectory[i + 1]

        num_states = len(active_states)

        attractors = []

        while num_states > 0:
            initial_state = active_states[0]

            rm_idx = np.where(active_states == initial_state)[0]
            active_states = np.delete(active_states, rm_idx)
            attractor_states = [initial_state]

            curr_state = transition[initial_state]
            while curr_state != initial_state:
                attractor_states.append(curr_state)

                rm_idx = np.where(active_states == curr_state)[0]
                active_states = np.delete(active_states, rm_idx)

                curr_state = transition[curr_state]

            attractors.append(attractor_states)
            num_states = len(active_states)

        return attractors

    def select_nodes(self, nodes: List[int]):
        new_F = list()
        new_varF = list()
        new_nv = list()
        for F_func, F_vars, n_vars in zip(self.F, self.varFInt, self.nv):
            # assumes F contains truthtables for sorted vars
            # print("F_vars ", F_vars)
            new_nv.append(0)
            new_varF.append(list())
            new_F.append(list())
            curr_num_vars = len(F_vars)
            curr_F = F_func
            current_removed = 0
            for i, var in enumerate(F_vars):
                if var not in nodes:
                    curr_i = i - current_removed
                    var_state = 0
                    # indeces = [j + (2**curr_i) * (j // (2**curr_i)) + (curr_i + 1) * var_state for j in range(2**(curr_num_vars - 1))]
                    # print("indeces - ", indeces, " var_state ", var_state, " curr_i ", curr_i)
                    curr_F = [
                        curr_F[j + (2**curr_i) * (j // (2**curr_i)) + (curr_i + 1) * var_state]
                        for j in range(2 ** (curr_num_vars - 1))
                    ]
                    curr_num_vars -= 1
                    current_removed += 1
                else:
                    new_varF[-1].append(var)
                    new_nv[-1] += 1

            new_F[-1].append(curr_F)

        # Translation of old nodes into new nodes
        translation = {node: idx for idx, node in enumerate(nodes)}

        def translate(to_translate: List[int]):
            return [translation[elem] for elem in to_translate]

        # Assumes nodes are numbered from 0 to n-1!!!!!
        n = len(nodes)
        nf = [self.nf[idx] for idx, i in enumerate(self.nf) if idx in nodes]
        nv = [new_nv[idx] for idx, i in enumerate(new_nv) if idx in nodes]
        F = [sublist[0] for sublist in new_F]
        F = [F[idx] for idx, i in enumerate(F) if idx in nodes]
        print()
        varFInt = [translate(new_varF[idx]) for idx, i in enumerate(new_varF) if idx in nodes]
        cij = self.cij
        perturbation = self.perturbation
        npNode = [np for np in self.npNode if np in nodes]
        npNode.append(n)

        return PBN(
            n, nf, nv, F, varFInt, cij, perturbation, npNode, self.n_parallel, update_type_int=1
        )

    # def segment_attractor(self, attractor_states, history):
    #     active_states = attractor_states
    #     transition = dict()

    #     for trajectory in history:
    #         for i in range(len(trajectory) - 1):
    #             if trajectory[i] in transition:
    #                 if transition[trajectory[i]] != trajectory[i + 1]:
    #                     raise ValueError("Two different states from the same state")

    #             transition[trajectory[i]] = trajectory[i + 1]

    #     num_states = len(active_states)

    #     attractors = []

    #     while num_states > 0:
    #         initial_state = active_states[0]

    #         rm_idx = np.where(active_states == initial_state)[0]
    #         active_states = np.delete(active_states, rm_idx)
    #         attractor_states = [initial_state]

    #         curr_state = transition[initial_state]
    #         while curr_state != initial_state:
    #             attractor_states.append(curr_state)

    #             rm_idx = np.where(active_states == curr_state)[0]
    #             active_states = np.delete(active_states, rm_idx)

    #             curr_state = transition[curr_state]

    #         attractors.append(attractor_states)
    #         num_states = len(active_states)

    #     return attractors

    def dependency_graph(self, filename: str | None = None) -> graphviz.Digraph:
        """
        Plot the dependency graph of a Probabilistic Boolean Network (PBN).

        This function creates a directed graph where each node represents a variable in the PBN,
        and each edge represents a dependency between variables. An edge from node $i$ to node $j$
        indicates that the value of $i$ influences the value of $j$.

        :param filename: The filename to save the graph as a PNG image. If None, the graph is not saved.
        :type filename: str, optional

        :return: A graphviz.Digraph object representing the dependency graph.
        :rtype: graphviz.Digraph
        """

        return bang.visualization.draw_dependencies(self, filename)

    def trajectory_graph(
        self,
        index: int,
        filename: str | None = None,
        format: Literal["pdf", "png", "svg"] = "svg",
        show_labels: bool = True,
    ) -> graphviz.Digraph:
        """
        Plot the trajectory of a Probabilistic Boolean Network (PBN).

        This function creates a directed graph where each node represents a state in the trajectory,
        and each edge represents a transition between states.

        :param index: The index of the trajectory to plot.
        :type index: int

        :param filename: The filename to save the graph. If None, the graph is not saved.
        :type filename: str, optional

        :param format: The format to save the graph in. Default is 'svg'.
        :type format: Literal['pdf', 'png', 'svg']

        :param show_labels: Whether to show labels on the nodes. Default is True. If set to False, the nodes are represented as points.
        :type show_labels: bool

        :return: A graphviz.Digraph object representing the trajectory graph.
        :rtype: graphviz.Digraph
        """

        return bang.visualization.draw_trajectory_ndarray(
            self.get_trajectories()[:, index, :], filename, format, show_labels
        )

    def block_graph(
        self, filename: str | None = None, format: Literal["pdf", "png", "svg"] = "svg"
    ) -> graphviz.Digraph:
        """
        Plot the blocks of a Probabilistic Boolean Network (PBN).

        This function creates a directed graph where each node represents a block in the PBN,
        and each edge represents a transition between blocks.

        :param filename: The filename to save the graph. If None, the graph is not saved.
        :type filename: str, optional

        :param format: The format to save the graph in. Default is 'svg'.
        :type format: Literal['pdf', 'png', 'svg']

        :return: A graphviz.Digraph object representing the block graph.
        :rtype: graphviz.Digraph
        """

        return bang.visualization.draw_blocks(self, filename, format)


def load_sbml(path: str) -> tuple:
    return parseSBMLDocument(path)


def load_from_file(path: str, format: str = "sbml") -> PBN:
    """
    Loads a PBN from files of format .pbn or .sbml.

    :param path: Path to the file of format .pbn.
    :type path: str
    :param format: Choose the format. Can be either 'sbml' for files with .sbml format or 'assa' for files with .pbn format. Defaults to 'sbml'.
    :type format: str, optional
    :returns: PBN object representing the network from the file.
    :rtype: PBN
    :raises ValueError: If the format is invalid.
    """
    match format:
        case "sbml":
            return PBN(*load_sbml(path))
        case "assa":
            return PBN(*load_assa(path))
        case _:
            raise ValueError("Invalid format")
