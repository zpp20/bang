import bang

pbn = bang.load_from_file("examples/test2_no_perturbation.pbn", "assa", n_parallel=7)

attractors = pbn.blocks_detect_attractors()

print(attractors)
