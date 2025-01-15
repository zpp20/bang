import bang

pbn = bang.load_from_file("tests/test2.pbn", "assa")
bang.german_gpu_run(pbn, 3, 1000)