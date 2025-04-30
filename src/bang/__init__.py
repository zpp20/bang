from numba import config

from .core import PBN, load_from_file
from .core.attractors.blocks.graph import get_blocks

config.CUDA_ENABLE_PYNVJITLINK = 1  # type: ignore

__all__ = ["PBN", "load_from_file", "get_blocks"]
