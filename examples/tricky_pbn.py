import bang

pbn1 = bang.PBN(
    2,
    [1, 1],
    [2, 2],
    [[True, True, True, False], [True, False, True, True]],
    [[0, 1], [0, 1]],
    [[1.0], [1.0]],
    0.0,
    [2],
    n_parallel=1,
)
pbn1.set_states([[False, False]])
pbn1.simple_steps(1)
print(pbn1.get_last_state())
