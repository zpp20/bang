"""
Module containing the PBN class and helpers.
"""

import math
from typing import Literal, overload

import graphviz
import numpy as np
import numpy.typing as npt
from numba import cuda

from bang.core.attractors.blocks.attractors_paralel import divide_and_counquer_gpu
from bang.core.attractors.blocks.divide_and_conquer import divide_and_conquer
from bang.core.attractors.blocks.graph import get_blocks
from bang.core.attractors.monolithic.monolithic import monolithic_detect_attractor
from bang.core.attractors.monte_carlo.monte_carlo import monte_carlo
from bang.core.pbn.array_management import GpuMemoryContainer
from bang.core.pbn.simple_steps import invoke_cpu_simulation, invoke_cuda_simulation
from bang.core.pbn.truthtable_reduction import reduce_F
from bang.core.pbn.utils.integer_functions import get_integer_functions
from bang.core.pbn.utils.state_printing import (
    convert_from_binary_representation,
    convert_to_binary_representation,
)
from bang.parsing.assa import load_assa
from bang.parsing.sbml import parseSBMLDocument
from bang.visualization import draw_blocks, draw_dependencies, draw_trajectory_ndarray

UpdateType = Literal["asynchronous_random_order", "asynchronous_one_random", "synchronous"]
DEFAULT_STEPS_BATCH_SIZE = 100000


class PBN:
    """Class representing the PBN and the execution of simulations on it.

    :param n_nodes: The number of nodes.
    :type n_nodes: int
    :param n_functions: The size of each node.
    :type n_functions: list[int]
    :param n_variables: The size of each node's truth table.
    :type n_variables: list[int]
    :param functions: The truth table of each node.
    :type functions: list[list[bool]]
    :param parent_variable_indices: The index of each node's truth table.
    :type parent_variable_indices: list[list[int]]
    :param function_probabilities: The selection probability of each function.
    :type function_probabilities: list[list[float]]
    :param perturbation_rate: The probability of perturbation at each step.
    :type perturbation_rate: float
    :param non_perturbed_nodes: Index of nodes without perturbation.
    :type non_perturbed_nodes: list[int]
    :param n_parallel: The number of parallel simulations. Defaults to 512.
    :type n_parallel: int, optional
    :param history: The execution history of the PBN, tracking the states of all trajectories.
    :type history: np.ndarray
    :param latest_state: The last encountered state of the PBN's trajectories.
    :type latest_state: np.ndarray
    :param previous_simulations: list of previous simulations.
    :type previous_simulations: list[np.ndarray]
    :param update_type: The type of update to use. The possible values are "asynchronous_one_random", "asynchronous_random_order", "synchronous"
    :type update_type: str
    :param save_history: Whether to save the history of the PBN.
    :type save_history: bool
    :param steps_batch_size: The size of the batch of the maximum number of steps executed in a single kernel invocation.
    :type steps_batch_size: int
    """

    def __init__(
        self,
        n_nodes: int,
        n_functions: list[int],
        n_variables: list[int],
        functions: list[list[bool]],
        parent_variable_indices: list[list[int]],
        function_probabilities: list[list[float]],
        perturbation_rate: float,
        non_perturbed_nodes: list[int],
        n_parallel: int = 512,
        update_type: UpdateType = "asynchronous_one_random",  ## TODO change to 0, synchronous shouldnt be default
        save_history: bool = True,
        steps_batch_size=DEFAULT_STEPS_BATCH_SIZE,
    ):
        # internal attributes
        self._n = n_nodes
        self._nf = n_functions
        self._nv = n_variables
        self._f = functions
        self._var_f_int = parent_variable_indices
        self._cij = function_probabilities
        self._perturbation = perturbation_rate
        self._np_node = list(sorted(non_perturbed_nodes))
        self._n_parallel = n_parallel
        self._history: np.ndarray = np.zeros((1, n_parallel, self.state_size), dtype=np.uint32)
        self._latest_state: np.ndarray = np.zeros((n_parallel, self.state_size), dtype=np.uint32)
        self._previous_simulations: list[np.ndarray] = []
        self._gpu_memory_container = None

        # public attributes
        self.update_type: UpdateType = update_type
        self.save_history = save_history
        self.steps_batch_size = steps_batch_size

        if cuda.is_available():
            self._create_memory_container()

    def clone_with(
        self,
        n: int | None = None,
        nf: list[int] | None = None,
        nv: list[int] | None = None,
        F: list[list[bool]] | None = None,
        varFInt: list[list[int]] | None = None,
        cij: list[list[float]] | None = None,
        perturbation: float | None = None,
        npNode: list[int] | None = None,
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
        :type nf: list[int], optional
        :param nv: The size of each node's truth table.
        :type nv: list[int], optional
        :param F: The truth table of each node.
        :type F: list[list[bool]], optional
        :param varFInt: The index of each node's truth table.
        :type varFInt: list[list[int]], optional
        :param cij: The selection probability of each node.
        :type cij: list[list[float]], optional
        :param perturbation: The perturbation rate.
        :type perturbation: float, optional
        :param npNode: Index of nodes without perturbation.
        :type npNode: list[int], optional
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
            n_nodes=n if n is not None else self._n,
            n_functions=nf if nf is not None else self._nf,
            n_variables=nv if nv is not None else self._nv,
            functions=F if F is not None else self._f,
            parent_variable_indices=varFInt if varFInt is not None else self._var_f_int,
            function_probabilities=cij if cij is not None else self._cij,
            perturbation_rate=perturbation if perturbation is not None else self._perturbation,
            non_perturbed_nodes=npNode if npNode is not None else self._np_node,
            n_parallel=n_parallel if n_parallel is not None else self._n_parallel,
            update_type=update_type if update_type is not None else self.update_type,
            save_history=save_history if save_history is not None else self.save_history,
            steps_batch_size=steps_batch_size
            if steps_batch_size is not None
            else self.steps_batch_size,
        )

    def _create_memory_container(self, stream=None):
        if stream is None:
            stream = cuda.default_stream()

        self._gpu_memory_container = GpuMemoryContainer(
            self, self.steps_batch_size, self.save_history, stream
        )

    def __str__(self):
        return f"PBN(n={self._n}, nf={self._nf}, nv={self._nv}, F={self._f}, varFInt={self._var_f_int}, cij={self._cij}, perturbation={self._perturbation}, npNode={self._np_node})"

    @property
    def n_nodes(self) -> int:
        """
        Returns the number of nodes.

        :returns: The number of nodes.
        :rtype: int
        """
        return self._n

    @property
    def n_functions(self) -> list[int]:
        """
        Returns the size of each node.

        :returns: The size of each node.
        :rtype: list[int]
        """
        return self._nf

    @property
    def n_variables(self) -> list[int]:
        """
        Returns the size of each node's truth table.

        :returns: The size of each node's truth table.
        :rtype: list[int]
        """
        return self._nv

    @property
    def functions(self) -> list[list[bool]]:
        """
        Returns the truth table of each node.

        :returns: The truth table of each node.
        :rtype: list[list[bool]]
        """
        return self._f

    @property
    def n_extra_functions(self) -> int:
        """
        Returns the number of extra functions.

        :returns: The number of extra functions.
        :rtype: int
        """
        extraFCount = 0
        for elem in self._nv:
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

        for elem in self._nv:
            if elem > 5:
                extraFIndexCount += 1

        return extraFIndexCount

    @property
    def extra_function_index(self) -> list[int]:
        """
        Returns a list of extra function indices.

        :returns: list of extra function indices.
        :rtype: list[int]
        """
        extraFIndex = []

        for i in range(len(self._nv)):
            if self._nv[i] > 5:
                extraFIndex.append(i)

        return extraFIndex

    @property
    def cum_extra_functions(self) -> list[int]:
        """
        Returns a list of cumulative extra functions.

        :returns: list of cumulative extra functions.
        :rtype: list[int]
        """
        cumExtraF = [0]

        for i in range(len(self._nv)):
            if self._nv[i] > 5:
                cumExtraF.append(cumExtraF[-1] + 2 ** (self._nv[i] - 5) - 1)

        return cumExtraF

    @property
    def extra_functions(self) -> list[int]:
        """
        Returns a list of extra functions.

        :returns: list of extra functions.
        :rtype: list[int]
        """
        extraF = []

        for i in range(len(self._f)):
            extraF.append(get_integer_functions(self._f[i], extraF))

        return extraF

    @property
    def cum_n_functions(self) -> np.ndarray:
        """
        Returns the cumulative sum of all elements in nf.

        :returns: Cumulative sum of all elements in nf.
        :rtype: np.ndarray
        """
        return np.cumsum([0] + self._nf)

    @property
    def cum_n_variables(self) -> np.ndarray:
        """
        Returns the cumulative sum of all elements in nv.

        :returns: Cumulative sum of all elements in nv.
        :rtype: np.ndarray
        """
        return np.cumsum([0] + self._nv)

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
        return self._calc_state_size(self._n)

    @property
    def integer_functions(self) -> list[int]:
        """
        Returns the integer representation of the truth table for each node.

        :returns: Integer representation of the truth table for each node.
        :rtype: list[int]
        """
        return [get_integer_functions(func, []) for func in self._f]

    @property
    def parent_variable_indices(self) -> list[list[int]]:
        """
        Returns the indices of the parent nodes for every boolean function.

        :returns: The indices of the parent nodes by function.
        :rtype: list[list[int]]
        """
        return self._var_f_int

    @property
    def function_probabilities(self) -> list[list[float]]:
        """
        Returns the selection probability of each function per node.

        :returns: The selection probability of each function per node.
        :rtype: list[list[float]]
        """
        return self._cij

    @property
    def perturbation_rate(self) -> float:
        """
        Returns the perturbation rate.

        :returns: The perturbation rate.
        :rtype: float
        """
        return self._perturbation

    @property
    def non_perturbed_nodes(self) -> list[int]:
        """
        Returns the indices of nodes without perturbation.

        :returns: The indices of nodes without perturbation.
        :rtype: list[int]
        """
        return self._np_node

    @property
    def last_state(self) -> np.ndarray:
        """
        Returns the last encountered state of the PBN's trajectories.

        :returns: The last encountered state of the PBN's trajectories.
        :rtype: np.ndarray
        """
        return self._latest_state

    @property
    def last_state_bool(self) -> list[list[bool]]:
        """
        Returns the last encountered state of the PBN's trajectories in boolean representation.

        :returns: The last encountered state of the PBN's trajectories in boolean representation.
        :rtype: list[list[bool]]
        """
        return convert_to_binary_representation(self._latest_state, self.n_nodes)

    @property
    def history(self) -> np.ndarray:
        """
        Returns the execution history of the PBN, tracking the states of all trajectories.

        :returns: The execution history of the PBN.
        :rtype: np.ndarray
        """
        return self._history

    @property
    def previous_simulations(self) -> list[np.ndarray]:
        """
        Returns the list of previous simulation histories. A new simulation history is added
        every time the number of trajectories changes.

        :returns: The list of previous simulations.
        :rtype: list[np.ndarray]
        """
        return self._previous_simulations

    @property
    def history_bool(self) -> list[list[list[bool]]]:
        """
        Returns the execution history of the PBN in boolean representation.

        :returns: The execution history of the PBN in boolean representation.
        :rtype: list[list[list[bool]]]
        """
        return convert_to_binary_representation(self._history, self.n_nodes)

    def save_trajectories(self, filename: str):
        """
        Saves the execution history of the PBN to a CSV file.

        :param filename: The name of the file to save the history.
        :type filename: str
        """
        np.save(filename, self._history)

    def save_last_state(self, filename: str):
        """
        Saves the last encountered state of the PBN's trajectories to a CSV file.

        :param filename: The name of the file to save the last state.
        :type filename: str
        """
        np.save(filename, self._latest_state)

    # Only for typing information
    @overload
    def get_blocks(self, repr: Literal["int"]) -> list[list[int]]:
        pass

    @overload
    def get_blocks(self, repr: Literal["bool"]) -> list[list[list[bool]]]:
        pass

    @overload
    def get_blocks(self) -> list[list[list[bool]]]:
        pass

    def get_blocks(self, repr="bool"):
        """
        Returns the blocks of the PBN.

        :param repr: The representation type. Can be "bool" or "int". Defaults to "bool".
        :type repr: str, optional

        :returns: The blocks of the PBN.
        :rtype: list[list[bool]] or list[list[int]]
        """
        blocks = get_blocks(self)

        if repr == "bool":
            return convert_to_binary_representation(blocks, self.n_nodes)
        elif repr == "int":
            return blocks
        else:
            raise ValueError("Invalid representation type. Use 'bool' or 'int'.")

    @staticmethod
    def _bools_to_state_array(bools: list[bool], node_count: int) -> np.ndarray:
        """
        Converts list of bools to integer array.

        :param bools: list of boolean values representing the state.
        :type bools: list[bool]
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

    def set_states(
        self,
        states: list[list[bool]] | npt.NDArray[np.uint32],
        reset_history: bool = False,
        stream=None,
    ):
        """
        Sets the initial states of the PBN. If the number of trajectories is different than the number of previous trajectories,
        the history will be pushed into `self.previous_simulations` and the active history will be reset.

        :param states: list of states to be set.
        :type states: list[list[bool]]
        :param reset_history: If True, the history of the PBN will be reset. Defaults to False.
        :type reset_history: bool, optional
        """
        converted_states = (
            [self._bools_to_state_array(state, self._n) for state in states]
            if isinstance(states, list)
            else states
        )

        self._n_parallel = len(states)
        self._latest_state = np.array(converted_states).reshape((self._n_parallel, self.state_size))

        if reset_history:
            self._history = np.array(converted_states).reshape(
                (1, self._n_parallel, self.state_size)
            )
        else:
            if len(states) != self._history.shape[1]:
                self._previous_simulations.append(self._history.copy())

                self._history = np.array(converted_states).reshape(
                    (1, self._n_parallel, self.state_size)
                )
            else:
                self._history = np.concatenate(
                    [
                        self._history,
                        np.array(converted_states).reshape((1, self._n_parallel, self.state_size)),
                    ],
                    axis=0,
                )

        if cuda.is_available():
            if stream is None:
                stream = cuda.default_stream()

            self._create_memory_container(stream=stream)

    def reduce_truthtables(self, states: list[list[int]]) -> tuple:
        """
        Reduces truth tables of PBN by removing states that do not change.

        :param states: list of investigated states. States are lists of int with length n where i-th index represents i-th variable. 0 represents False and 1 represents True.
        :type states: list[list[int]]
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
                for _ in range(n_steps // self.steps_batch_size):
                    invoke_cuda_simulation(self, self.steps_batch_size, actions)

                if n_steps % self.steps_batch_size != 0:
                    invoke_cuda_simulation(self, n_steps % self.steps_batch_size, actions)
            else:
                invoke_cuda_simulation(self, n_steps, actions)
        elif device == "cuda" and not cuda.is_available():
            print("WARNING! CUDA is not available, falling back to CPU simulation")
            invoke_cpu_simulation(self, n_steps, actions)
        else:
            invoke_cpu_simulation(self, n_steps, actions)

    # typing only
    @overload
    def monolithic_detect_attractors(
        self, initial_states, repr: Literal["bool"]
    ) -> list[list[list[bool]]]:
        pass

    @overload
    def monolithic_detect_attractors(self, initial_states, repr: Literal["int"]) -> list[list[int]]:
        pass

    @overload
    def monolithic_detect_attractors(self, initial_states) -> list[list[list[bool]]]:
        pass

    def monolithic_detect_attractors(self, initial_states, repr="bool"):
        """
        Detects all atractor states in PBN

        Parameters
        ----------

        initial_states : list[list[Bool]]
            list of investigated states.
        Returns
        -------
        attractor_states : list[list[int]]
            list of attractors where attractors are coded as lists of ints, ints representing the states.
        """

        attractors = monolithic_detect_attractor(self, initial_states)

        if repr == "bool":
            return convert_to_binary_representation(attractors, self.n_nodes)
        elif repr == "int":
            return attractors
        else:
            raise ValueError("Invalid representation type. Use 'bool' or 'int'.")

    # typing only
    @overload
    def blocks_detect_attractors(self, repr: Literal["int"]) -> list[list[int]]:
        pass

    @overload
    def blocks_detect_attractors(self, repr: Literal["bool"]) -> list[list[list[bool]]]:
        pass

    def blocks_detect_attractors(self, repr="bool"):
        """
        Detects attractors in the system using a divide-and-conquer block-based approach.

        Returns
        -------
        attractor_states : list[list[list[bool]]] or list[list[int]]
            list of attractors where attractors are coded as lists of lists of bools, lists of bools representing the states.
        """
        attractors = divide_and_conquer(self)

        if repr == "bool":
            return attractors
        elif repr == "int":
            return convert_from_binary_representation(attractors)
        else:
            raise ValueError("Invalid representation type. Use 'bool' or 'int'.")

    def blocks_detect_attractors_parallel(self) -> list[npt.NDArray[np.uint32]]:
        """
        Detects attractors in the system using a divide-and-conquer block-based approach parallelized on the cpu as well as gpu.

        Returns
        -------
        attractor_states : numpy.NDArray[np.uint32]
            list of attractors where attractors are coded as 2D lists od 32 bit unsigned integers.
        """

        return divide_and_counquer_gpu(self)  # type: ignore

    def monte_carlo_detect_attractors(
        self, trajectory_length: int, attractor_length: int, repr="bool"
    ):
        """
        Detects attractors in the system by running multiple trajectories and checking for repetitions.

        Parameters
        ----------

        trajectory_length : int
            Length after which we assume each trajectory is in attractor.

        initial_trajectory_length : int, optional
            Length of trajectory from which we read attractors.

        Returns
        -------
        attractor_states : list[list[list[bool]]] or list[list[int]]
            list of attractors where attractors are coded as lists of lists of bools, lists of bools representing the states.

        """
        attractors = monte_carlo(self, trajectory_length, attractor_length)

        if repr == "int":
            return attractors
        elif repr == "bool":
            return convert_to_binary_representation(attractors, self._n)
        else:
            raise ValueError("Invalid representation type. Use 'bool' or 'int'.")

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

        return draw_trajectory_ndarray(self.history[:, index, :], filename, format, show_labels)

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


def load_from_file(path: str, format: str = "sbml", n_parallel=512) -> PBN:
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
            return PBN(*load_sbml(path), n_parallel=n_parallel)
        case "assa":
            return PBN(*load_assa(path), n_parallel=n_parallel)
        case _:
            raise ValueError("Invalid format")
