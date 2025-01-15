import bang

<<<<<<< Updated upstream
pbn = bang.load_from_file("tests/test2.pbn", "assa")
bang.initialise_PBN(pbn)
bang.german_gpu_run()
=======
pbn = bang.load_from_file("test2.pbn", "assa")

print(pbn)

pbn.simple_steps(1000)
>>>>>>> Stashed changes
