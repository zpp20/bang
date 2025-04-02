import bang

initial_states = [[True, True],[False, True],[True, False],[False, False]]

n = 2
F = [[True, True, True, False],[False, True, False, False]]
varFInt = [[0,1],[0,1]]
nf = [1,1]
nv = [2,2]
n_parallel = 4
cij = [[1.],[1.]]
perturbation = 0.
npNode = [2]

pbn = bang.PBN(n,nf,nv,F,varFInt,cij,perturbation,npNode,n_parallel)

pbn.set_states(initial_states, reset_history=True)

pbn.simple_steps(1)

print(pbn.get_last_state())