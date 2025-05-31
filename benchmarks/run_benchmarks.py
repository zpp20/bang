import csv
import itertools
import re
import subprocess

# Grid of parameters to benchmark
update_types = ["synchronous", "asynchronous_one_random", "asynchronous_random_order"]
n_steps_list = [1000, 5000, 10000]
n_steps_list = [100, 1000]
save_history = [False]
n_nodes_list = [16, 32, 64, 128, 256]
n_nodes_list = [64]
n_parallels = [64, 256, 1024, 2048, 4096, 8192, 16384]

script_path = "bencher.py"
output_csv = "benchmark_results.csv"

# For resuming from a specific point e.g. after a crash or interruption
# start_from = {
#     "update_type": "asynchronous_one_random",
#     "n_steps": 1000,
#     "save_history": True,
#     "n_nodes": 32,
#     "n_parallel": 1024,
# }

start_from = None

start_found = False if start_from is not None else True

# Regex to capture elapsed time
elapsed_time_pattern = re.compile(r"Elapsed kernel execution time:\s*([\d.]+)\s*ms")

with open(output_csv, mode="w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        ["update_type", "n_steps", "save_history", "n_nodes", "n_parallel", "elapsed_time_ms"]
    )

    param_combinations = itertools.product(
        update_types, n_steps_list, save_history, n_nodes_list, n_parallels
    )

    for update_type, n_steps, save_history, n_nodes, n_parallel in param_combinations:
        # If a start_from is defined, skip until we match
        if start_from is not None and not start_found:
            if (
                update_type == start_from["update_type"]
                and n_steps == start_from["n_steps"]
                and save_history == start_from["save_history"]
                and n_nodes == start_from["n_nodes"]
                and n_parallel == start_from["n_parallel"]
            ):
                start_found = True
            else:
                continue

        cmd = [
            "python",
            script_path,
            "--update_type",
            update_type,
            "--n_steps",
            str(n_steps),
            "--n_nodes",
            str(n_nodes),
            "--n_parallel",
            str(n_parallel),
            "--cpu",
        ]

        if save_history:
            cmd += ["--save_history"]

        retry = True
        n_retries = 0
        elapsed_time = 0

        while retry and n_retries < 5:
            print(f"Running retry {n_retries}: {cmd}")

            result = subprocess.run(cmd, capture_output=True, text=True)

            stdout = result.stdout
            stderr = result.stderr

            m = elapsed_time_pattern.findall(stdout)

            if m:
                retry = False

                time_start_index = 0 if "--cpu" in cmd else 1

                for time in m[time_start_index:]:
                    elapsed_time += float(time)

                print(f"Total elapsed time found: {elapsed_time:.3f} ms")
                print(f"stdout:\n{stdout}")

            else:
                print(f"Warning: No elapsed time found for parameters: {cmd}")
                print(f"stdout:\n{stdout}")
                print(f"stderr:\n{stderr}")

            n_retries += 1

        # Write results to CSV
        csvwriter.writerow([update_type, n_steps, save_history, n_nodes, n_parallel, elapsed_time])

print("Benchmarking complete")
