import bang

n = 3
nf = [1, 1, 1]
nv = [1, 3, 1]
#f_1(x_3) = -x_1, f_2(x_1, x_2, x_3) = x_2 \and (x_1 \or x_3), f_3(x_1) = x_3
F = [[False, True], [False, False, False, True, False, False, True, True], [True, False]]
varFInt = [[0], [0, 1, 2] ,[2]]
cij = [[1], [1],[1]]
perturbation = 0.001
npNode = [2]

pbn = bang.PBN(n, nf, nv, F, varFInt, cij, perturbation, npNode)

active, reduced_F = pbn.reduce_F([[0,1,0],[0,1,1]])

print(active)
print(reduced_F)
