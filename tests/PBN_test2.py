import bang

pbn = bang.load_from_file("tests/test2.pbn", "assa")
print(pbn.cij)
print(pbn.F)
bang.initialise_PBN(pbn)
bang.german_gpu_run()