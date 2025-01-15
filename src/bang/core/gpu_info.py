from pycuda import driver as drv

def get_gpu_info():
    
    print("Reading GPU info...")
    

    device = drv.Device(0)

    major = device.COMPUTE_CAPABILITY_MAJOR
    minor = device.COMPUTE_CAPABILITY_MINOR
    SM_count = device.MULTIPROCESSOR_COUNT
    max_shmem_per_block = device.MAX_SHARED_MEMORY_PER_BLOCK
    register_per_SM = device.MAX_REGISTERS_PER_MULTIPROCESSOR
    max_blocks_per_SM = device.MAX_BLOCKS_PER_MULTIPROCESSOR

    return (major, minor, SM_count, max_shmem_per_block, register_per_SM, max_blocks_per_SM)

   