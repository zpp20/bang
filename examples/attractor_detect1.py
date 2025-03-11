import bang

#x_2 <- -(x_1 or x_2) and x_1
#x_1 <- x_2
n = 2
nf = [1,1]
nv = [1,2]
F = [[True, False],[False, False, True, True]]
varFInt = [[1],[0,1]]
cij = [[1.],[1.]]
perturbation = 0.
npNode = [0,1,2]
n_parallel = 3

pbn = bang.PBN(n,nf,nv,F,varFInt,cij,perturbation,npNode,n_parallel)

initial_states = [[False, False], [True, False], [False, True]]

attractor = pbn.detect_attractor(initial_states)

print(attractor)