PBN class documentation
=======================

.. class:: PBN

    Class representing Probabilistic Boolean Networks.

    This class stores information about a Probabilistic Boolean Networks, parses informations about PBNs from files and formats data for simulation. 

    :ivar int n: number of nodes.
    :ivar list[int] nf: number of boolean functions for each node.
    :ivar list[int] nv: number of variables for each boolean function.
    :ivar list[list[bool]] F: truth table of each boolean function.
    :ivar list[list[int]] varFInt: number of node each variable of boolean function references
    :ivar list[list[float]] cij: probability of each boolean function to be picked.
    :ivar float perturbation: perturbation rate.
    :ivar list[int] npNode: index of nodes without perturbation

    Methods:
    --------

    .. py:function:: load_from_file(path, format='sbml') -> PBN
    
        Loads file of a given format from path and creates object of class PBN from it.
        (sbml files are still WIP)

        :param path: path to the file.
        :type path: str
        :param format: format of the file. Can be 'sbml' for .sbml files or 'assa' for .pbn files. Devaults to '.sbml'
        :type format: str
