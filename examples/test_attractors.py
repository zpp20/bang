from bang.core.attractors.blocks.divide_and_conquer import divide_and_conquer
from bang.core.pbn.pbn import PBN

SCP_PBN = PBN(
    3,
    [1, 1, 1],
    [2, 2, 1],
    [[True, True, True, False], [False, True, False, False], [True, False]],
    [[0, 1], [0, 1], [1]],
    [[1.0], [1.0], [1.0]],
    0.0,
    [3],
)
SCP_PBN2 = PBN(
    5,
    [1, 1, 1, 1, 1],
    [2, 2, 1, 3, 2],
    [
        [True, True, True, False],
        [False, True, False, False],
        [True, False],
        [False, False, False, True, True, True, True, True],
        [False, True, True, True],
    ],
    [[0, 1], [0, 1], [1], [1, 2, 4], [3, 4]],
    [[1.0], [1.0], [1.0], [1.0], [1.0]],
    0.0,
    [3],
)
SCP_PBN3 = PBN(
    6,
    [1, 1, 1, 1, 1, 1],
    [2, 2, 1, 3, 2, 2],
    [
        [True, True, True, False],
        [False, True, False, False],
        [True, False],
        [False, False, False, True, True, True, True, True],
        [False, True, True, True],
        [False, False, True, False],
    ],
    [[0, 1], [0, 1], [1], [1, 2, 4], [3, 4], [2, 5]],
    [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
    0.0,
    [3],
)

print(divide_and_conquer(SCP_PBN))
print(divide_and_conquer(SCP_PBN2))
print(divide_and_conquer(SCP_PBN3))
