"""
Module containing the PBN class and helpers.
"""

import math
from typing import List, Literal

import graphviz
import numpy as np
import numpy.typing as npt
from numba import cuda

from bang.core.attractors.blocks.divide_and_conquer import divide_and_conquer
from bang.core.attractors.blocks.graph import get_blocks
from bang.core.attractors.monolithic.monolithic import monolithic_detect_attractor
from bang.core.pbn.array_management import GpuMemoryContainer
from bang.core.pbn.simple_steps import invoke_cpu_simulation, invoke_cuda_simulation
from bang.core.pbn.truthtable_reduction import reduce_F
from bang.parsing.assa import load_assa
from bang.parsing.sbml import parseSBMLDocument
from bang.visualization import draw_blocks, draw_dependencies, draw_trajectory_ndarray

UpdateType = Literal["asynchronous_random_order", "asynchronous_one_random", "synchronous"]
DEFAULT_STEPS_BATCH_SIZE = 100000


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
    :param update_type: The type of update to use. The possible values are "asynchronous_one_random", "asynchronous_random_order", "synchronous"
    :type update_type: str
    :param save_history: Whether to save the history of the PBN.
    :type save_history: bool
    :param steps_batch_size: The size of the batch of the maximum number of steps executed in a single kernel invocation.
    :type steps_batch_size: int
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
        update_type: UpdateType = "asynchronous_one_random",  ## TODO change to 0, synchronous shouldnt be default
        save_history: bool = True,
        steps_batch_size=DEFAULT_STEPS_BATCH_SIZE,
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
        self.update_type: UpdateType = update_type
        self.history: np.ndarray = np.zeros((1, n_parallel, self.state_size), dtype=np.uint32)
        self.latest_state: np.ndarray = np.zeros((n_parallel, self.state_size), dtype=np.uint32)
        self.previous_simulations: List[np.ndarray] = []
        self.save_history = save_history
        self.gpu_memory_container = None
        self.steps_batch_size = steps_batch_size

        if cuda.is_available():
            self._create_memory_container()

    def clone_with(
        self,
        n: int | None = None,
        nf: List[int] | None = None,
        nv: List[int] | None = None,
        F: List[List[bool]] | None = None,
        varFInt: List[List[int]] | None = None,
        cij: List[List[float]] | None = None,
        perturbation: float | None = None,
        npNode: List[int] | None = None,
        n_parallel: int | None = None,
        update_type: UpdateType | None = None,
        save_history: bool | None = None,
        steps_batch_size: int | None = None,
    ) -> "PBN":
        """
        Creates a clone of the PBN with the specified parameters.
        If a parameter is not provided, the corresponding attribute of the original PBN is used.

        :param n: The number of nodes.
        :type n: int, optional
        :param nf: The size of each node.
        :type nf: List[int], optional
        :param nv: The size of each node's truth table.
        :type nv: List[int], optional
        :param F: The truth table of each node.
        :type F: List[List[bool]], optional
        :param varFInt: The index of each node's truth table.
        :type varFInt: List[List[int]], optional
        :param cij: The selection probability of each node.
        :type cij: List[List[float]], optional
        :param perturbation: The perturbation rate.
        :type perturbation: float, optional
        :param npNode: Index of nodes without perturbation.
        :type npNode: List[int], optional
        :param n_parallel: The number of parallel simulations.
        :type n_parallel: int, optional
        :param update_type: The type of update to use. The possible values are "asynchronous_one_random", "asynchronous_random_order", "synchronous"
        :type update_type: str, optional
        :param save_history: Whether to save the history of the PBN.
        :type save_history: bool, optional
        :param steps_batch_size: The size of the batch of the maximum number of steps executed in a single kernel invocation.
        :type steps_batch_size: int, optional
        :returns: A new PBN object with the specified parameters.
        :rtype: PBN
        """

        return PBN(
            n=n if n is not None else self.n,
            nf=nf if nf is not None else self.nf,
            nv=nv if nv is not None else self.nv,
            F=F if F is not None else self.F,
            varFInt=varFInt if varFInt is not None else self.varFInt,
            cij=cij if cij is not None else self.cij,
            perturbation=perturbation if perturbation is not None else self.perturbation,
            npNode=npNode if npNode is not None else self.npNode,
            n_parallel=n_parallel if n_parallel is not None else self.n_parallel,
            update_type=update_type if update_type is not None else self.update_type,
            save_history=save_history if save_history is not None else self.save_history,
            steps_batch_size=steps_batch_size if steps_batch_size is not None else self.steps_batch_size,
        )

    def _create_memory_container(self):
        self.gpu_memory_container = GpuMemoryContainer(
            self, DEFAULT_STEPS_BATCH_SIZE, self.save_history
        )

    def __str__(self):
        return f"PBN(n={self.n}, nf={self.nf}, nv={self.nv}, F={self.F}, varFInt={self.varFInt}, cij={self.cij}, perturbation={self.perturbation}, npNode={self.npNode})"

    @property
    def n_nodes(self) -> int:
        """
        Returns the number of nodes.

        :returns: The number of nodes.
        :rtype: int
        """
        return self.n

    @property
    def n_functions(self) -> List[int]:
        """
        Returns the size of each node.

        :returns: The size of each node.
        :rtype: List[int]
        """
        return self.nf

    @property
    def n_variables(self) -> List[int]:
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

    @property
    def last_state(self) -> np.ndarray:
        """
        Returns the last encountered state of the PBN's trajectories.

        :returns: The last encountered state of the PBN's trajectories.
        :rtype: np.ndarray
        """
        return self.latest_state

    @property
    def trajectories(self) -> np.ndarray:
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
        return get_blocks(self)

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

        integer_state = np.zeros((state_size), dtype=np.uint32)

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
        self.latest_state = np.array(converted_states).reshape((self.n_parallel, self.state_size))

        if reset_history:
            self.history = np.array(converted_states).reshape(
                (1, self.n_parallel, self.state_size)
            )
        else:
            if len(states) != self.history.shape[1]:
                self.previous_simulations.append(self.history.copy())

                self.history = np.array(converted_states).reshape(
                    (1, self.n_parallel, self.state_size)
                )
            else:
                self.history = np.concatenate(
                    [
                        self.history,
                        np.array(converted_states).reshape((1, self.n_parallel, self.state_size)),
                    ],
                    axis=0,
                )

        if cuda.is_available():
            self._create_memory_container()

    @property
    def n_extra_functions(self) -> int:
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

    @property
    def n_extra_function_index(self) -> int:
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

    @property
    def extra_function_index(self) -> List[int]:
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

    @property
    def cum_extra_functions(self) -> List[int]:
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

    @property
    def extra_functions(self) -> List[int]:
        """
        Returns a list of extra functions.

        :returns: List of extra functions.
        :rtype: List[int]
        """
        extraF = []

        for i in range(len(self.F)):
            extraF.append(self._get_integer_functions(self.F[i], extraF))

        return extraF

    @property
    def cum_n_functions(self) -> np.ndarray:
        """
        Returns the cumulative sum of all elements in nf.

        :returns: Cumulative sum of all elements in nf.
        :rtype: np.ndarray
        """
        return np.cumsum([0] + self.nf)

    @property
    def cum_n_variables(self) -> np.ndarray:
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

    @property
    def state_size(self) -> int:
        """
        Returns the number of 32-bit integers needed to store all variables.

        :returns: Number of 32-bit integers needed to store all variables.
        :rtype: int
        """
        return self._calc_state_size(self.n)

    @property
    def functions(self) -> List[List[bool]]:
        """
        Returns the truth table of each node.

        :returns: The truth table of each node.
        :rtype: List[List[bool]]
        """
        return self.F

    @property
    def integer_functions(self) -> List[int]:
        """
        Returns the integer representation of the truth table for each node.

        :returns: Integer representation of the truth table for each node.
        :rtype: List[int]
        """
        return [self._get_integer_functions(func, []) for func in self.F]

    @property
    def parent_variable_indices(self) -> List[List[int]]:
        """
        Returns the indices of the parent nodes for every boolean function.

        :returns: The indices of the parent nodes by function.
        :rtype: List[List[int]]
        """
        return self.varFInt

    @property
    def function_probabilities(self) -> List[List[float]]:
        """
        Returns the selection probability of each function per node.

        :returns: The selection probability of each function per node.
        :rtype: List[List[float]]
        """
        return self.cij

    @property
    def perturbation_rate(self) -> float:
        """
        Returns the perturbation rate.

        :returns: The perturbation rate.
        :rtype: float
        """
        return self.perturbation

    @property
    def non_perturbed_nodes(self) -> List[int]:
        """
        Returns the indices of nodes without perturbation.

        :returns: The indices of nodes without perturbation.
        :rtype: List[int]
        """
        return self.npNode

    def reduce_truthtables(self, states: List[List[int]]) -> tuple:
        """
        Reduces truth tables of PBN by removing states that do not change.

        :param states: List of investigated states. States are lists of int with length n where i-th index represents i-th variable. 0 represents False and 1 represents True.
        :type states: List[List[int]]
        :returns: Tuple containing list of indices of variables that change between states and truth tables with removed constant variables.
        :rtype: tuple
        """

        return reduce_F(self, states)

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

    def simple_steps(
        self,
        n_steps: int,
        actions: npt.NDArray[np.uint] | None = None,
        device: Literal["cuda", "cpu"] = "cuda",
    ):
        """
        Simulates the PBN for a given number of steps.

        :param n_steps: Number of steps to simulate.
        :type n_steps: int
        :param actions: Array of actions to be performed on the PBN. Defaults to None.
        :type actions: npt.NDArray[np.uint], optional
        :raises ValueError: If the initial state is not set before simulation.
        """
        if device == "cuda" and cuda.is_available():
            if self.save_history:
                # batch simple_step executions to avoid allocating too much memory for history
                for _ in range(n_steps // DEFAULT_STEPS_BATCH_SIZE):
                    invoke_cuda_simulation(self, DEFAULT_STEPS_BATCH_SIZE, actions)

                if n_steps % DEFAULT_STEPS_BATCH_SIZE != 0:
                    invoke_cuda_simulation(self, n_steps % DEFAULT_STEPS_BATCH_SIZE, actions)
            else:
                invoke_cuda_simulation(self, n_steps, actions)
        elif device == "cuda" and not cuda.is_available():
            print("WARNING! CUDA is not available, falling back to CPU simulation")
            invoke_cpu_simulation(self, n_steps, actions)
        else:
            invoke_cpu_simulation(self, n_steps, actions)

    def monolithic_detect_attractors(self, initial_states):
        """
        Detects all atractor states in PBN

        Parameters
        ----------

        initial_states : List[List[Bool]]
            List of investigated states.
        Returns
        -------
        attractor_states : list[list[int]]
            List of attractors where attractors are coded as lists of ints, ints representing the states.
        """

        return monolithic_detect_attractor(self, initial_states)

    def blocks_detect_attractors(self):
        """
        Detects attractors in the system using a divide-and-conquer block-based approach.

        Returns
        -------
        attractor_states : list[list[list[bool]]]
            List of attractors where attractors are coded as lists of lists of bools, lists of bools representing the states.
        """

        return divide_and_conquer(self)

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

        return draw_dependencies(self, filename)

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

        return draw_trajectory_ndarray(
            self.trajectories[:, index, :], filename, format, show_labels
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

        return draw_blocks(self, filename, format)


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
