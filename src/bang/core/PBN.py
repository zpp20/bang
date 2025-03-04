from typing import List
import os
from math import floor
import numpy as np
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

    def _bools_to_state_array(self, bools: list[bool]) -> np.ndarray:
        """
        Converts list of bools to integer
        """
        if len(bools) != self.n:
            raise ValueError("Number of bools must be equal to number of nodes")

        integer_state = np.zeros((self.stateSize()), dtype=np.int32)

        for i in range(self.stateSize())[::-1]:
            for bit in range(32):
                if bools[i * 32 + bit]:
                    integer_state[i] |= 1 << bit

        return integer_state

    def set_states(self, states: list[list[bool]]):
        converted_states = [self._bools_to_state_array(state) for state in states]
        self.latest_state = np.array(converted_states)

        self.history = None

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

    def stateSize(self):
        """
        Returns number of 32 bit integers needed to store all variables
        """
        return self.n // 32 + math.ceil(self.n / 32)

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
                    indeces = [j + (2**curr_i) * (j // (2**curr_i)) + curr_i * var_state for j in range(2**(curr_num_vars - 1))]
                    # print("indeces - ", indeces)
                    curr_F = [curr_F[j + (2**curr_i) * (j // (2**curr_i)) + curr_i * var_state] for j in range(2**(curr_num_vars - 1))]
                    curr_num_vars -= 1
                    current_removed += 1
                else:
                    new_varF[-1].append(var)

            new_F[-1].append(curr_F)

        return new_varF, new_F




