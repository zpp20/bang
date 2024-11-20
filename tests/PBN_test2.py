import bang

pbn = bang.load_from_file("test2.pbn", "assa")
bang.initialise_PBN(pbn)
bang.german_gpu_run()