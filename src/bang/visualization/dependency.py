import graphviz
from bang.core import PBN
from typing import Literal

def draw_dependencies(pbn: PBN, filename: str | None = None, format: Literal['pdf', 'png', 'svg'] = 'svg') -> graphviz.Digraph:
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

    dot.attr('node', shape='circle')

    for i in range(pbn.n):
        dot.node(str(i), label=f"{i}")

    for i in range(pbn.n):
        for j in range(pbn.nf[i]):
            dot.edge(str(i), str(pbn.varFInt[i][j]))

    if filename is not None:
        dot.render(filename, format, cleanup=True)

    return dot