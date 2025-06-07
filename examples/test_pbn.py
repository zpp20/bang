import bang

example_bn = bang.PBN(
    6,  # Number of nodes of PBN
    [1, 1, 1, 1, 1, 1],  # Number of Boolean functions per node
    [2, 2, 1, 3, 2, 2],  # Number of variables per boolean function
    [  # Truth tables representing each function
        [True, True, True, False],
        [False, True, False, False],
        [True, False],
        [False, False, False, True, True, True, True, True],
        [False, True, True, True],
        [False, False, True, False],
    ],
    [[0, 1], [0, 1], [1], [1, 2, 4], [3, 4], [2, 5]],  # Variables acting on each Boolean function
    [
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
    ],  # Probability of picking each function for different nodes (always one since we have BN)
    0.0,  # Probability of perturbation (0.0 since we have BN)
    [7],  # Nodes for which we dont perform perturbation (irrelevant since we have BN)
    update_type="synchronous",  # BN updates synchronously. All nodes update at once.
)

example_bn.blocks_detect_attractors()
