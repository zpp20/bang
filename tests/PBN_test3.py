import bang

pbn = bang.load_from_file("tests/test2.pbn", "assa")
bang.initialise_PBN(pbn)
bang.simple_step(3)