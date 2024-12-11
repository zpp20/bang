Functions related to simulation
===============================

.. py:function:: initialise_PBN(pbn: PBN)

    Initializes PBN in global variables for simulation. Must be run before any type of simulation.

    :param pbn: Probabilistic Boolean Network to be used in simulation
    :type pbn: PBN

.. py:function:: german_gpu_run()

    Simulates 5 steps on multiple copies of PBN and writes out final states to stdout.