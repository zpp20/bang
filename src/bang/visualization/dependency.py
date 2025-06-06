import typing
from typing import Literal

import graphviz
import numpy as np

if typing.TYPE_CHECKING:
    from bang.core import PBN


def draw_dependencies(
    pbn: "PBN",
    filename: str | None = None,
    format: Literal["pdf", "png", "svg"] = "svg",
    number_from_one: bool = False,
) -> graphviz.Digraph:
    """
    Plot the dependency graph of a Probabilistic Boolean Network (PBN).

    This function creates a directed graph where each node represents a variable in the PBN,
    and each edge represents a dependency between variables.

    :param pbn: The Probabilistic Boolean Network (PBN) to plot.
    :type pbn: PBN

    :param filename: The filename to save the graph. If None, the graph is not saved.
    :type filename: str, optional

    :param format: The format to save the graph in. Default is 'svg'.
    :type format: Literal['pdf', 'png', 'svg']

    :return: A graphviz.Digraph object representing the dependency graph.
    :rtype: graphviz.Digraph
    """
    dot = graphviz.Digraph()

    dot.attr("node", shape="circle")

    for i in range(pbn._n):
        label = f"{i + 1}" if number_from_one else f"{i}"
        dot.node(str(i), label)

    cumNf = np.cumsum([0] + pbn._nf, dtype=np.int32)

    edges = set()

    for node_ind in range(pbn._n):
        for func_ind in range(pbn._nf[node_ind]):
            f_index = cumNf[node_ind] + func_ind
            f_parents = pbn._var_f_int[f_index]

            for parent in f_parents:
                if (parent, node_ind) not in edges:
                    dot.edge(str(parent), str(node_ind))
                    edges.add((parent, node_ind))

    if filename is not None:
        dot.render(filename, format, cleanup=True)

    return dot
