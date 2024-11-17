import bang

n = 2
nf = [1,1]
nv = [1,1]
F = [[True, False], [False, True]]
varFInt = [[1],[0]]
cij = [[1],[1]]
perturbation = 0.01
npNode = [2]

pbn = bang.PBN(n, nf, nv, F, varFInt, cij, perturbation, npNode)

bang.initialise_PBN(pbn)

bang.german_gpu_run()