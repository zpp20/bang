from lxml import etree
import bang

PBN = bang.load_from_file("tests/test3.sbml")
bang.initialise_PBN(PBN)
bang.german_gpu_run()

