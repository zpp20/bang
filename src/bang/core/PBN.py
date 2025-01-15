from typing import List
import numpy as np
import math
from itertools import chain
from bang.core.cuda.simulation import kernel_converge
from numba import cuda

from bang.parsing.assa import load_assa
from bang.parsing.sbml import parseSBMLDocument

from numba.cuda.random import create_xoroshiro128p_states


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
    ):

        self.n = n  # the number of nodes
        self.nf = nf  # the size is n
        self.nv = nv  # the sizef = cum(nf)
        self.F = F  # each element of F stores a column of the truth table "F"， e.g., F.get(0)=[true false], the length of the element is 2^nv(boolean function index)
        self.varFInt = varFInt
        self.cij = cij  # the size=n, each element represents the selection probability of a node, and therefore the size of each element equals to the number of functions of each node
        self.perturbation = perturbation  # perturbation rate
        self.npNode = npNode  # index of those nodes without perturbation. To make simulation easy, the last element of npNode will always be n, which also indicate the end of the list. If there is only one element, it means there is no disabled node.

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

    # def getFInfo(self):
    #     """
    #     Returns list of size 6:
    #     - getFInfo[0] is extraFCount
    #     - getFInfo[1] is extraFIndexCount
    #     - getFInfo[2] is list extraFIndex of size extraFIndexCount
    #     - getFInfo[3] is list cumExtraF of size extraFIndexCount + 1
    #     - getFInfo[4] is list extraF of size extraFCount
    #     - getFInfo[5] is list F of size cumnF

    #     In case there are no extraFs extraFCount, extraFIndexCount are set to 1.
    #     extraFIndex is list of ones of size 1
    #     extraF is list of ones of size 1
    #     cumExtraF is list of ones of size 2
    #     (in original implementation lists were allocated but no numbers were assigned)
    #     """
    #     extraFCount = 0
    #     extraFIndexCount = 0
    #     for elem in self.nv:
    #         if elem > 5:
    #             extraFIndexCount += 1
    #             extraFCount += 2**(elem - 5) - 1

    #     if extraFIndexCount > 0:
    #         extraFIndex = list()
    #         cumExtraF = list()
    #         extraF = list()

    #         cumExtraF.append(0)
    #         for i in range(len(self.nv)):
    #             if self.nv[i] > 5:
    #                 extraFIndex.append(i)
    #                 cumExtraF.append(cumExtraF[-1] + 2**(self.nv[i] - 5) - 1)

    #     else:
    #         extraFCount = 1
    #         extraFIndexCount = 1
    #         extraFIndex = [1]
    #         cumExtraF = [1,1]
    #         extraF = [1]

    #     F = list()

    #     for i in range(len(self.F)):
    #         F.append(self.getVector(self.F[i], extraF))

    #     return [extraFCount, extraFIndexCount, extraFIndex, cumExtraF, extraF, F]

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

    def simple_steps(self, n_steps):
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

        block = 1
        blockSize = 32

        N = block * blockSize

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

        states = create_xoroshiro128p_states(N, seed=time.time())

        kernel_converge[block, blockSize](
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
        history = gpu_stateHistory.copy_to_host()

        print(last_state)
        print(last_state.shape)
        print(history.reshape((n_steps + 1, -1)))


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
