import bang

pbn = bang.load_from_file("examples/test2_no_perturbation.pbn", "assa", n_parallel=7)

attr = pbn.monte_carlo_detect_attractors(trajectory_length= 200, minimum_repetitions=5, initial_trajectory_length=5, repr='bool')

print(attr)


attr = pbn.monte_carlo_detect_attractors(trajectory_length= 200, minimum_repetitions=5, initial_trajectory_length=5, repr='int')

print(attr)