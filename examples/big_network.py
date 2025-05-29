import bang

#this network has 7 nodes!
pbn = bang.load_from_file("examples/model.sbml", format='sbml', n_parallel=20)

# attractors = pbn.monte_carlo_detect_attractors(trajectory_length = 10000, attractor_length = 1000)

# print(attractors)
print(f"n - {pbn._n}")
print(f"len f - {len(pbn._f)}")