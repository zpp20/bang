import colorsys
import random
from typing import Literal

import graphviz
import numpy as np

from bang.core import PBN
from bang.graph.graph import get_blocks


def generate_contrasting_colors(n: int) -> list[str]:
    colors = []

    for i in range(n):
        h = i / n
        s = 0.7 + random.random() * 0.3  # Saturation between 0.7 and 1.0
        v = 0.7 + random.random() * 0.3  # Value between 0.7 and 1.0
        r, g, b = [int(256 * i) for i in colorsys.hsv_to_rgb(h, s, v)]
        colors.append("#{:02x}{:02x}{:02x}".format(r, g, b))

    return colors


def create_color_string(
    node: int, node_to_blocks: dict[int, set[int]], color_list: list[str]
) -> str:
    colors = [color_list[block] for block in node_to_blocks[node]]
    fraction = 1 / len(colors)

    scaled_colors = [f"{color};{fraction}" for color in colors]

    return ":".join(scaled_colors)


def draw_blocks(
    pbn: PBN, filename: str | None = None, format: Literal["pdf", "png", "svg"] = "svg"
) -> graphviz.Digraph:
    """
    Plot the blocks of a Probabilistic Boolean Network (PBN).

    This function creates a directed graph where each node represents a block in the PBN,
    and each edge represents a transition between blocks.

    :param pbn: The Probabilistic Boolean Network (PBN) to plot.
    :type pbn: PBN

    :param filename: The filename to save the graph. If None, the graph is not saved.
    :type filename: str, optional

    :param format: The format to save the graph in. Default is 'svg'.
    :type format: Literal['pdf', 'png', 'svg']

    :return: A graphviz.Digraph object representing the block graph.
    :rtype: graphviz.Digraph
    """
    blocks = get_blocks(pbn)

    node_to_blocks = {}

    contrasting_colors = generate_contrasting_colors(len(blocks))

    for block_ind, block in enumerate(blocks):
        for node in block:
            node_to_blocks[node] = set((block_ind,)) | node_to_blocks.get(node, set())

    dot = graphviz.Digraph()

    dot.attr("node", shape="circle")

    for i in range(pbn.n):
        dot.node(
            str(i),
            label=f"{i}",
            color=create_color_string(i, node_to_blocks, contrasting_colors),
            style="wedged",
        )

    cumNf = np.cumsum([0] + pbn.nf, dtype=np.int32)

    edges = set()

    for node_ind in range(pbn.n):
        for func_ind in range(pbn.nf[node_ind]):
            f_index = cumNf[node_ind] + func_ind
            f_parents = pbn.varFInt[f_index]

            for parent in f_parents:
                if (parent, node_ind) not in edges:
                    dot.edge(str(parent), str(node_ind))
                    edges.add((parent, node_ind))

    if filename is not None:
        dot.render(filename, format, cleanup=True)

    return dot
