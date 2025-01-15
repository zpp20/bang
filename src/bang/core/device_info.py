import pycuda.driver as drv
import bang

def _compute_device_info(shared_memory_size : int, 
                        state_size : int,
                        trajectory_length : int,
                        trajectory_num : int):
    """
    Function computing number of blocks and blockSize for kernel in a way that maximizes occupancy.
    """

    # major = device.COMPUTE_CAPABILITY_MAJOR
    # minor = device.COMPUTE_CAPABILITY_MINOR
    major = bang.major
    minor = bang.minor

    if (major < 3 or (major < 4 and minor < 2)):
        raise ValueError("Compute capability to small. Compute capability bigger than 3.0 required!")

    
    # SM_count = device.MULTIPROCESSOR_COUNT
    # max_shmem_per_block = device.MAX_SHARED_MEMORY_PER_BLOCK
    # register_per_SM = device.MAX_REGISTERS_PER_MULTIPROCESSOR
    # max_blocks_per_SM = device.MAX_BLOCKS_PER_MULTIPROCESSOR
    SM_count = bang.SM_count
    max_shmem_per_block = bang.max_shmem_per_block
    register_per_SM = bang.register_per_SM
    max_blocks_per_SM = bang.max_blocks_per_SM
    
    warp_size = 32
    register_per_thread = 63                     #TODO: figure out how to compute this based on PBN stats

    if (shared_memory_size + state_size * trajectory_length > max_shmem_per_block):
        raise ValueError("The PBN is too large for the current device")

    trajectory_size = state_size * trajectory_length
    select_block_size = 32
    select_block_count = 1


    block_size = 32
    count_block_size = 1
    block_count = 1
    occupancy = 0
    possible = True
    # print("Trajectory size - ", trajectory_size)
    # print("Need ", trajectory_num, " trajectories")
    while (possible):
        # print("")
        # print("Block count - ", block_count)
        possible = False
        active_blocks_per_SM = block_count

        if active_blocks_per_SM > max_blocks_per_SM:
            active_blocks_per_SM = max_blocks_per_SM

        count_block_size = 1
        block_size = count_block_size * warp_size
        all_threads = block_size * block_count * SM_count
        size_shared_memory_trajectory = shared_memory_size + trajectory_size * block_size
        # print("Registers per SM - ", register_per_SM)
        # print("")
        # print("Max block size allowed by registers - ", (register_per_SM / block_count) / register_per_thread)
        # print("Shmem per block - ", max_shmem_per_block / block_count)
        # print("All threads - ", all_threads)
        while (block_size < (register_per_SM / block_count) / register_per_thread and
                size_shared_memory_trajectory < max_shmem_per_block / block_count and
                all_threads <= trajectory_num):
            # print("Block size - ", block_size)
            
            if max_shmem_per_block / block_count < active_blocks_per_SM:
                active_blocks_per_SM = max_shmem_per_block
            
            if (register_per_SM / register_per_thread / block_size) < active_blocks_per_SM:
                active_blocks_per_SM = register_per_SM / register_per_thread / block_size

            if (active_blocks_per_SM * block_size > occupancy or 
                (active_blocks_per_SM * block_size == occupancy and block_count * SM_count > select_block_count)):
                # print("Better occupancy: ", occupancy, " threads active and ", block_count * SM_count, " blocks active")
                occupancy = active_blocks_per_SM * block_size
                select_block_size = block_size
                select_block_count = block_count * SM_count
                
            possible = True
            count_block_size += 1
            block_size = count_block_size * warp_size
            all_threads = block_size * block_count * SM_count

        block_count += 1

    return (select_block_count * SM_count, select_block_size)
            