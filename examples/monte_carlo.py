import bang

pbn = bang.load_from_file("examples/test2_no_perturbation.pbn", "assa", n_parallel=7)

attractors = pbn.monte_carlo_detect_attractors(trajectory_length=10000, attractor_length=4)

print("MC ATTRACTORS: ")
for attractor in attractors:
    print(attractor)


print("DET ATTRACTORS: ")

det_attr = pbn.blocks_detect_attractors()

for attractor in det_attr:
    print(attractor)
