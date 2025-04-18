import bang

pbn = bang.load_from_file("examples/test2_no_perturbation.pbn", "assa", n_parallel=3)

attr = pbn.detect_attractors_monte_carlo(num_steps=1, step_length=50, repetitions=4)

print(attr)