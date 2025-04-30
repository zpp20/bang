from bang import PBN
from bang.core.attractors.blocks.node_selection import select_nodes

n = 2
nf = [1, 1]
nv = [2, 1]
F = [[True, False, False, True], [False, False]]
varFInt = [[0, 1], [1]]
cij = [[1.0], [1.0]]
perturbation = 0
npNode = [2]

pbn = PBN(n, nf, nv, F, varFInt, cij, perturbation, npNode)

new_pbn = select_nodes(pbn, [0])

print("new pbn truthtable - ", new_pbn.F)
print("new pbn varFInt - ", new_pbn.varFInt)
