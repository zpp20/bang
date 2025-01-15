import pycuda.driver as drv
from .core.gpu_info import get_gpu_info

global major, minor, SM_count, max_shmem_per_block, register_per_SM, max_blocks_per_SM

drv.init()
major, minor, SM_count, max_shmem_per_block, register_per_SM, max_blocks_per_SM = get_gpu_info()

from .core import PBN, load_from_file, german_gpu_run

__all__ = ['PBN', 'load_from_file', 'german_gpu_run']
