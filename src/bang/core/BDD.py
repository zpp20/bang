from typing import List
import numpy as np
import numpy.typing as npt
import numba
from numba import cuda
from bang.core.cuda.bdd_traverse import kernel_BDD_step

class BDD:
    #Every state in takes 63 bits. Youngest 5 bits code number of variable, next 29 bits code index of left child, next 29 bits code index of right child.
    def __init__(
        self,
        states : List[int],
        variables : List[int],
        n: int
    ):
        self.n = n
        self.states = states
        self.variables = variables

    def get_states(self):
        return np.array(self.states).astype(np.uint64)

    def get_variables(self):
        return np.array(self.variables).astype(np.uint32)

    def build_BDD(states : List[List[int]], variables : List[int]):
        n = len(variables)

        compressed_states = list()

        for node in states:
            compressed_node = 0
            mask_5 =  (1 << 5) - 1
            mask_29 = (1 << 29) - 1

            compressed_node &= ~mask_5
            compressed_node |= (node[0] & mask_5)
            # print("-----")
            # print(compressed_node) 
            compressed_node &= ~(mask_29 << 5)
            compressed_node |= (node[1] & mask_29) << 5
            # print(compressed_node)            
            compressed_node &= ~(mask_29 << 34)
            compressed_node |= (node[2] & mask_29) << 34
            # print(f"{compressed_node:b}") 
            compressed_states.append(compressed_node)

        return BDD(compressed_states, variables, n)

def traverse_BDD(BDDs: List[BDD], initial_states: npt.NDArray[np.uint64], n_states : int):
    if len(BDDs) != n_states:
        raise ValueError("Number of BDDs must be equal to number of states!")

    BDD_states = [bdd.get_states() for bdd in BDDs]
    BDD_variables = [bdd.get_variables() for bdd in BDDs]

    n_states = [len(bdd.get_states()) for bdd in BDDs]
    n_variables = [len(bdd.get_variables()) for bdd in BDDs]

    #Starting indeces for states and variables in 1d BDD array
    cum_n_states = np.cumsum([0] + n_states, dtype=np.uint32)
    cum_n_variables = np.cumsum([0] + n_variables, dtype=np.uint32)
    print(cum_n_states)
    print(cum_n_variables)
    flat_BDD_states = np.concatenate(BDD_states).astype(np.uint64)
    flat_BDD_variables = np.concatenate(BDD_variables).astype(np.uint32)

    gpu_BDD_states = cuda.to_device(np.array(flat_BDD_states, dtype=np.uint64))
    gpu_BDD_variables = cuda.to_device(np.array(flat_BDD_variables, dtype=np.uint64))
    gpu_cum_n_states = cuda.to_device(np.array(cum_n_states, dtype=np.uint64))
    gpu_cum_n_variables = cuda.to_device(np.array(cum_n_variables, dtype=np.uint64))
    gpu_BDD_initial_states = cuda.to_device(np.array(initial_states, dtype=np.uint64))
    
        
