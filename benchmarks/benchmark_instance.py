import argparse
import bang
from numba import cuda

import time

def main():
    # config.DISABLE_JIT = True
    # os.environ["NUMBA_DISABLE_JIT"] = "1"

    parser = argparse.ArgumentParser(description="Run a BANG PBN simulation with configurable parameters.")
    parser.add_argument("--update_type", type=str, default="synchronous", help="Type of update (e.g., synchronous, asynchronous)")
    parser.add_argument("--n_steps", type=int, default=10000, help="Number of steps to simulate")
    parser.add_argument("--save_history", help="Whether to save history", action="store_true")
    parser.add_argument("--n_nodes", type=int, default=256, help="Number of nodes in the network")
    parser.add_argument("--n_parallel", type=int, default=1024 * 8, help="Number of parallel simulations")
    parser.add_argument("--cpu", help="Whether to conduct the benchmark on the CPU instead of GPU", action="store_true")

    args = parser.parse_args()

    print("Running PBN Simulation with the following parameters:")
    print(f"  Update type        : {args.update_type}")
    print(f"  Number of steps    : {args.n_steps}")
    print(f"  Save history       : {args.save_history}")
    print(f"  Number of nodes    : {args.n_nodes}")
    print(f"  Number of parallels: {args.n_parallel}")
    print(f"  CPU execution: {args.cpu}")
    print("-" * 50)

    test_n_nodes = args.n_nodes

    pbn_num_func = [1 for _ in range(test_n_nodes)]
    pbn_num_var = [2 for _ in range(test_n_nodes)]
    pbn_probs = [[1.0] for _ in range(test_n_nodes)]
    pbn_var_indexes = [[i // 2, i // 2 + 1] for i in range(test_n_nodes)]
    functions_1 = [[True, True, True, False] for _ in range(test_n_nodes // 2)]
    functions_2 = [[True, False, True, True] for _ in range(test_n_nodes // 2)]
    pbn_functions: list[list[bool]] = []

    for a, b in zip(functions_1, functions_2):
        pbn_functions.append(a)
        pbn_functions.append(b)

    pbn2 = bang.PBN(
        test_n_nodes,
        pbn_num_func,
        pbn_num_var,
        pbn_functions,
        pbn_var_indexes,
        pbn_probs,
        0.0,
        [test_n_nodes],
        n_parallel=args.n_parallel,
        update_type=args.update_type,
        save_history=args.save_history,
        steps_batch_size=50000
    )

    print("shape state history", pbn2.history.shape)
    print("save history", pbn2.save_history)



    if args.cpu:
        # dry run
        pbn2.simple_steps(args.n_steps, device='cpu')

        start = time.process_time()
        pbn2.simple_steps(args.n_steps, device='cpu')
        end = time.process_time()

        print("Elapsed kernel execution time:", (end - start) * 1000, "ms")
    else:
        # dry run - numba may compile first and prolong the measurement
        pbn2.simple_steps(args.n_steps)
        pbn2.simple_steps(args.n_steps)



    print("\nSimulation complete.")

    cuda.close()

if __name__ == "__main__":
    main()
