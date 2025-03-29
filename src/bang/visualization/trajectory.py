from typing import Literal

import graphviz
import numpy as np


def draw_trajectory_ndarray(
    trajectory: np.ndarray,
    filename: str | None = None,
    format: Literal["pdf", "png", "svg"] = "svg",
    show_labels: bool = True,
) -> graphviz.Digraph:
    """
    Plot the trajectory of a Probabilistic Boolean Network (PBN).

    This function creates a directed graph where each node represents a state in the trajectory,
    and each edge represents a transition between states.

    :param trajectory: The trajectory to plot, where each row represents a state.
    :type trajectory: np.ndarray

    :param filename: The filename to save the graph. If None, the graph is not saved.
    :type filename: str, optional

    :param format: The format to save the graph in. Default is 'svg'.
    :type format: Literal['pdf', 'png', 'svg']

    :param show_labels: Whether to show labels on the nodes. Default is True. If set to False, the nodes are represented as points.
    :type show_labels: bool

    :return: A graphviz.Digraph object representing the trajectory graph.
    :rtype: graphviz.Digraph
    """
    dot = graphviz.Digraph()

    edges: set[tuple[str, str]] = set()

    if show_labels:
        dot.attr("node", shape="circle")
    else:
        dot.attr("node", shape="point")

    dot.attr("edge", arrowsize="0.5")

    nodes = set()

    nodes.add(str(trajectory[0, :]))
    dot.node(str(trajectory[0, :]), str(trajectory[0, :]))

    for prev, next in zip(trajectory[:-1, :], trajectory[1:, :]):
        if str(next) not in nodes:
            dot.node(str(next), str(next))

        if (str(prev), str(next)) not in edges:
            dot.edge(str(prev), str(next))
            edges.add((str(prev), str(next)))

    if filename is not None:
        dot.render(filename, format, cleanup=True)

    return dot
