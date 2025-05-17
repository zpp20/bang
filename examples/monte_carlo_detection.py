import bang

pbn = bang.load_from_file("examples/test2_no_perturbation.pbn", "assa", n_parallel=7)

attr_deter = pbn.blocks_detect_attractors(repr="bool")

print(attr_deter)

attr = pbn.monte_carlo_detect_attractors(trajectory_length= 200, attractor_length=100, repr='bool')

print(attr)


attr = pbn.monte_carlo_detect_attractors(trajectory_length= 200, attractor_length=100, repr='int')

print(attr)