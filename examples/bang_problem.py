from typing import TypeAlias, TypeVar, Type

import os
import boolean as bool
from sympy.logic.boolalg import truth_table
import bang

# Type definitions
BNReal = TypeVar("BNReal", bound="BN_Realisation")
State: TypeAlias = tuple[int, ...]


def int2bin(x,num_nodes):
    return format(x,'0'+str(num_nodes)+'b')


class BN_Realisation:
    __bool_algebra = bool.BooleanAlgebra()


    """
    Reads node names and their associated Boolean functions from a file in the ISPL format.

    Args:
        path_to_ispl_file (str): The path to the Boolean network model specification file in ISPL format.

    Returns:
        BN_Realisation: A Boolean network object containing the nodes and Boolean functions 
            specified in the ISPL file.
    """
    @classmethod
    def load_ispl(cls: Type[BNReal], path_to_ispl_file: str) -> BNReal:
        if not os.path.isfile(path_to_ispl_file):
            raise FileNotFoundError(path_to_ispl_file)

        BN_variables = []
        BN_functions = []

        with open(path_to_ispl_file, "r") as ispl_file:
            line = ispl_file.readline()
            while line:
                while line.strip() != "Vars:":
                    line = ispl_file.readline()

                line = ispl_file.readline()

                while line.strip() != "end Vars":
                    line = line.strip()
                    gene_name = line.split(':')[0]
                    BN_variables.append(gene_name.strip())
                    line = ispl_file.readline()

                while line.strip() != "Evolution:":
                    line = ispl_file.readline()

                line = ispl_file.readline()

                while line.strip() != "end Evolution":
                    line = ispl_file.readline()
                    line = line.strip()
                    line = line.split(" if ")[1]
                    line = line.split("=")[0]
                    BN_functions.append(line.strip())
                    line = ispl_file.readline()

                while line.strip() != "InitStates":
                    line = ispl_file.readline()
                break

        assert len(BN_variables) == len(BN_functions), "The number of nodes does not match the number of Boolean functions."

        print(f"Loaded a Boolean network of {len(BN_variables)} nodes.")

        return BN_Realisation(BN_variables, BN_functions)


    """
    BN_Realisation class constructore - initializes a BN_Realisation object.

    Args:
        list_of_nodes (list[str]): A list of node names in the Boolean network.
        list_of_functions (list[str]): A list of Boolean functions in the network. 
            The order of the functions must correspond to the order of the nodes.
        mode (str): The update scheme for the Boolean network. Must be either 
            'asynchronous' or 'synchronous'.

    Returns:
        BN_Realisation: An instance of the BN_Realisation class.
    """
    def __init__(self,
                 list_of_nodes: list[str],
                 list_of_functions: list[str],
                 mode: str = "asynchronous") -> BNReal:

        if mode not in ['asynchronous', 'synchronous']:
            raise ValueError(f"Wrong update scheme: {mode}")

        # Mode of the model.
        self.mode = mode

        # Number of nodes.
        self.num_nodes = len(list_of_nodes)

        # Names of nodes.
        self.node_names = list_of_nodes

        # String representing the update rules.
        self.functions_str = list_of_functions

        # Holds bool_algebra.Symbols of nodes.
        self.list_of_nodes = []

        for node_name in list_of_nodes:
            node = self.__bool_algebra.Symbol(node_name)
            self.list_of_nodes.append(node)

        self.functions = []
        for fun in list_of_functions:
            self.functions.append(self.__bool_algebra.parse(fun, simplify=True))



    def simulate(self, nsteps: int, init_state: State = None):
        var_indices = {var: i for i, var in enumerate(self.node_names)}
        parent_variables = [sorted([var_indices[var.__str__()] for var in f.symbols]) for f in self.functions]
        truth_tables = [[y for _, y in truth_table(self.functions_str[i], [x.__str__() for x in self.functions[i].symbols])] for
                        i in range(self.num_nodes)]

        pbn = bang.PBN(self.num_nodes,
                       [1 for _ in range(self.num_nodes)],
                       [len(f.symbols) for f in self.functions],
                       truth_tables,
                       parent_variables,
                       [[1.] for _ in range(self.num_nodes)],
                       0.,
                       [],
                       n_parallel=min(max(77, self.num_nodes * 10), 2 ** self.num_nodes - 1))
        pbn._n_parallel = min(max(77, pbn.n_nodes * 10), 2 ** pbn.n_nodes - 1)
        pbn.device = "gpu"

        if init_state is not None:

            pbn.set_states([[True if bit==1 else False for bit in init_state]])

        else:

            init_state = [random.choices([True, False], k=self.num_nodes)]
            pbn.set_states(init_state)

        # Simulate the network for nsteps
        pbn.simple_steps(nsteps, device='cpu')

        trajectory = [int2bin(s[0][0], self.num_nodes)[::-1] for s in pbn.history]

        return trajectory


bn = BN_Realisation.load_ispl('examples/bn_7.ispl')
print(bn.simulate(nsteps=10, init_state=(0,0,0,0,0,0,1)))