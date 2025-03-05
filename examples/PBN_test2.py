import bang

pbn = bang.load_from_file("examples/test2.pbn", "assa")

print(pbn)

pbn.simple_steps(1000)