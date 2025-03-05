from bang.core.BDD import BDD, traverse_BDD
import numpy as np

simple_BDD_2 = [
    [0,1,2],
    [1,3,4],
    [31,0,0],
    [31,0,0],
    [30,0,0]
]

simple_BDD_1 = [
    [1,1,2],
    [30,0,0],
    [31,0,0]
]

#BDDs for PBN: 00->01->10, 10->01, 11 (attractors (01,10), (11))

bdd1 = BDD.build_BDD(simple_BDD_1, [0,1])
bdd2 = BDD.build_BDD(simple_BDD_2, [0,1])

initial_states = np.array([0b00, 0b01, 0b10], dtype=np.uint64)

for state in bdd2.states:
    print(f"{state:b}")

# traverse_BDD([bdd1, bdd2], initial_states, 2)

