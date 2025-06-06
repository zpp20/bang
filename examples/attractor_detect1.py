import bang

# x_2 <- -(x_1 or x_2) and x_1
# x_1 <- x_2
n = 2
nf = [1, 1]
nv = [1, 1]
F = [[False, True], [True, False]]
varFInt = [[0], [1]]
cij = [[1.0], [1.0]]
perturbation = 0.0
npNode = [0, 1, 2]
n_parallel = 4

pbn = bang.PBN(n, nf, nv, F, varFInt, cij, perturbation, npNode, n_parallel)

initial_states = [[False, False], [True, False], [False, True], [True, True]]

attractors = pbn.monolithic_detect_attractors(initial_states)

print(attractors)
