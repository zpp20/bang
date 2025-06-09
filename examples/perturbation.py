import bang

pbn = bang.load_from_file("examples/example_network.pbn", 'assa')

pbn.update_type = "synchronous"

pbn.set_states([[True, True, True] for _ in range(1)], reset_history=True)
pbn.simple_steps(5)

for i in range(len(pbn.history_bool)):
    print(pbn.history_bool[i][0])