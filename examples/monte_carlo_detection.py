import bang

pbn = bang.load_from_file("examples/test2.pbn", "assa")

attr = pbn.detect_attractors_monte_carlo(num_steps=1, step_length=14, repetitions=2)

print(attr)