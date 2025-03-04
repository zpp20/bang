from numba import cuda
import numba as nb

@cuda.jit
def kernel_BDD_step(
    gpu_BDD_states,
    gpu_num_states,
    gpu_BDD_variables,
    gpu_cum_n_states,
    gpu_cum_n_variables,
    gpu_BDD_initial_states,
    gpu_num_active_states,
    gpu_current_BDD,
    gpu_current_indeces,
    gpu_terminal_states
):
    idx = cuda.grid(1)
    
    #if more threads than active states then exit
    if idx >= gpu_num_active_states[0]:
        return

    #get number of states and currently processed BDD
    num_states = gpu_num_states[0]
    current_BDD = gpu_current_BDD[idx]
    current_BDD_base_idx = gpu_cum_n_states[current_BDD]

    #get processed variable index
    current_BDD_state = gpu_BDD_states[current_BDD_base_idx + gpu_current_indeces[idx]]

    current_variable_idx = current_BDD_state & 0b11111

    if curr_variable_idx > 29:  #states do not need to change, we are in 0 if 30 and 1 if 31 so we exit
        return  

    #get index of variable that corresponds to node
    current_variable = gpu_BDD_variables[gpu_cum_n_states[current_BDD] + current_variable_idx]

    #get state of current variable. either 0 or 1
    current_state = (gpu_BDD_initial_states[current_BDD] >> current_variable) & 1
    
    #select mask for child
    base_mask = (1 << 29) - 1
    zero_child_mask = base_mask << 6
    one_child_mask = base_mask << 35

    mask = zero_child_mask & ~(-current_state) | (one_child_mask & -current_state)

    next_idx = current_BDD_state & mask

    gpu_current_indeces[idx] = next_idx
    


