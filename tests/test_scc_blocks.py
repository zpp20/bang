from bang.core import PBN
from bang.graph.graph import Graph_PBN


def setup_pbn():
    f1 = [False in range(48)] + [True in range(16)]
    f2 = [False in range(32)] + [True in range(16)] + [False in range(16)]
    f3 = [True in range(16)] + [False in range(16)] + [True in range(16)] + [False in range(16)]

    f4 = []
    for k in range(2):
        for i in range(3):
            for j in range(2):
                f4.append(False)
            for j in range(2):
                f4.append(True)
            for j in range(2):
                f4.append(False)
            for j in range(2):
                f4.append(True)
        for i in range(8):
            f4.append(True)

    f5 = []
    for i in range(8):
        for j in range(2):
            f5.append(False)
        for j in range(6):
            f5.append(True)

    f6 = []

    for i in range(2):
        for j in range(4):
            f6.append(False)
        for j in range(4):
            f6.append(True)
        for j in range(8):
            f6.append(False)
        for j in range(4):
            f6.append(False)
        for j in range(4):
            f6.append(True)
        for j in range(8):
            f6.append(False)

    # This is the example from the paper of dr Mizera, page 10 on the left
    return PBN(
        6,
        [1, 1, 1, 1, 1, 1],
        [3, 2, 1, 2, 2, 2],
        [f4, f2, f3, f1, f5, f6],
        [[1, 2, 4], [3, 1], [1], [3, 1], [0, 4], [2, 5]],
        [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
        0.01,
        [2, 3, 4, 5],
    )


def test_should_find_sccs_in_small_pbn():
    pbn = PBN(
        2, [1, 1], [1, 1], [[True, False], [False, True]], [[1], [0]], [[1.0], [1.0]], 0.01, [2]
    )

    graph = Graph_PBN(pbn)
    graph.find_scc_and_blocks()

    assert graph.sccs == [[0, 1]]


def test_should_find_blocks_in_small_pbn():
    pbn = PBN(
        2, [1, 1], [1, 1], [[True, False], [False, True]], [[1], [0]], [[1.0], [1.0]], 0.01, [2]
    )

    graph = Graph_PBN(pbn)
    graph.find_scc_and_blocks()

    assert graph.blocks == [[0, 1]]


def test_should_find_sccs_in_big_pbn():
    pbn = setup_pbn()

    graph = Graph_PBN(pbn)
    graph.find_scc_and_blocks()

    assert graph.sccs == [[0, 4], [1, 3], [2], [5]]


def test_should_find_blocks():
    pbn = setup_pbn()

    graph = Graph_PBN(pbn)
    graph.find_scc_and_blocks()

    assert graph.blocks == [[0, 1, 2, 4], [1, 3], [1, 2], [2, 5]]
