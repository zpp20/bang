from typing import List
from libsbml import *
import itertools
import os
#from parseSBML import parseSBMLDocument

class PBN:

    def __init__(self, n : int,
                nf : List[int], 
                nv : List[int],
                F : List[List[bool]],
                varFInt : List[List[int]],
                cij : List[List[float]],
                perturbation : float,
                npNode : List[int]):
        
        self.n = n  #the number of nodes
        self.nf = nf    #the size is n
        self.nv = nv    #the sizef = cum(nf)
        self.F = F  #each element of F stores a column of the truth table "F"， e.g., F.get(0)=[true false], the length of the element is 2^nv(boolean function index)
        self.varFInt = varFInt  
        self.cij = cij  #the size=n, each element represents the selection probability of a node, and therefore the size of each element equals to the number of functions of each node
        self.perturbation = perturbation    #perturbation rate
        self.npNode = npNode    #index of those nodes without perturbation. To make simulation easy, the last element of npNode will always be n, which also indicate the end of the list. If there is only one element, it means there is no disabled node.


    def getN(self):
        return self.n

    def getNf(self):
        return self.nf
    
    def getNv(self):
        return self.nv
    
    def getVector(self, elemF, extraF):
        """
        based on fromVector function from original ASSA-PBN
        converts list of bools to 32 bit int with bits representing truth table
        extra bits are placed into extraF list
        """
        
        retval = 0
        i = 0
        prefix = 0
        tempLen = len(elemF)
        if tempLen > 32:                                                       #we have to add values to extraF
            for i in range(32):
                if elemF[i + prefix]:
                    retval |= 1 << i

            prefix += 32
            tempLen -= 32
        
        else:                                                                  #we just return proper into
            for i in range(tempLen):
                if elemF[i]:
                    retval |= 1 << i
            
            return retval
        
        while tempLen > 0:                                                     #switched condition to tempLen > 0 to get one more iteration after tempLen > 32 is false
            other = 0
            for i in range(32):
                if elemF[i + prefix]:
                    other |= 1 << i
            
            prefix += 32
            tempLen -= 32
            extraF.append(other)

        return retval
        
    def getFInfo(self):
        """
        Returns list of size 6: 
        - getFInfo[0] is extraFCount
        - getFInfo[1] is extraFIndexCount
        - getFInfo[2] is list extraFIndex of size extraFIndexCount
        - getFInfo[3] is list cumExtraF of size extraFIndexCount + 1
        - getFInfo[4] is list extraF of size extraFCount
        - getFInfo[5] is list F of size cumnF
        
        In case there are no extraFs extraFCount, extraFIndexCount are set to 1.
        extraFIndex is list of ones of size 1
        extraF is list of ones of size 1
        cumExtraF is list of ones of size 2
        (in original implementation lists were allocated but no numbers were assigned)
        """
        extraFCount = 0
        extraFIndexCount = 0
        for elem in self.nv:
            if elem > 5:
                extraFIndexCount += 1
                extraFCount += 2**(elem - 5) - 1

        if extraFIndexCount > 0:
            extraFIndex = list()
            cumExtraF = list()
            extraF = list()

            cumExtraF.append(0)
            for i in range(len(self.nv)):
                if self.nv[i] > 5:
                    extraFIndex.append(i)
                    cumExtraF.append(cumExtraF[-1] + 2**(self.nv[i] - 5) - 1)
                
        else:
            extraFCount = 1
            extraFIndexCount = 1
            extraFIndex = [1]
            cumExtraF = [1,1]
            extraF = [1]

        F = list()

        for i in range(len(self.F)):
            F.append(self.getVector(self.F[i], extraF))

        return [extraFCount, extraFIndexCount, extraFIndex, cumExtraF, extraF, F]
             

    def getF(self):
        return self.F
    
    def getVarFInt(self):
        return self.varFInt
    
    def getCij(self):
        return self.cij
    
    def getPerturbation(self):
        return self.perturbation
    
    def getNpNode(self):
        return self.npNode
    



def load_sbml(path: str) -> PBN:
    #return PBN(*parseSBMLDocument(path))
    pass

def load_assa(path):
    n = 0
    nf = []
    nv = []
    F = []
    varFInt = []
    cij = []
    perturbation = 0.0
    np = []
    type = ''
    i = 0

    # forbidden characters in the variable names
    forbidden_chars = [' ', '\t', '\n', '\r', '\v', '\f', '&', '*', '!', '^', '/', '|', ':', '(', ')']
    # forbidden variable names
    forbidden_names = ['or', 'not', 'and']
    # for python eval function
    logical_replacements = {
        '|': ' or ',
        '&': ' and ',
        '!': ' not '
    }

    names_dict = {}
    index_dict = {}
 
    def delete_comments_and_empty_lines(lines):
        new_lines = []
        for line in lines:
            line = line.strip()
            if line == "" or line.startswith("//"):
                continue
            new_lines.append(line)
        return new_lines

    # extract variable names in the 'fun' function expression  in ASSA format
    def get_vars_from_assa_expr(fun):
        vars = []
        cur = ''
        started = False
        for i in range(len(fun)):
            # if the character is a forbidden, the variable name is finished
            if fun[i] in forbidden_chars:
                if started:
                    if cur not in vars:
                        if cur not in names_dict:
                            raise ValueError("Invalid file format")
                        vars.append(cur)
                    cur = ''
                    started = False
                else:
                    continue
            else:
                cur += fun[i]
                started = True
        if started:
            if cur not in vars:
                if cur in forbidden_names or cur not in names_dict:
                     raise ValueError("Invalid file format")
                vars.append(cur)
        return vars
    
    def get_n(line):
        nonlocal n
        line = line.strip()
        if line.startswith("n=") == False:
            raise ValueError("Invalid file format")
        n = line.split("=")[1]
        if not n.isnumeric():
            raise ValueError("Invalid file format")
        n = int(n)
        nonlocal i
        i += 1

    def get_type(line):
        nonlocal type
        line = line.strip()
        if line.startswith("type=") == False:
            raise ValueError("Invalid file format")
        type = line.split("=")[1]
        if type not in ['synchronous', 'rog', 'rmg', 'rmgrm', 'rmgro', 'rmgrorm', 'aro']:
            raise ValueError("Invalid file format")
        nonlocal i
        i += 1
    
    def get_perturbation(line):
        nonlocal perturbation
        line = line.strip()
        if line.startswith("perturbation=") == False:
            raise ValueError("Invalid file format")
        perturbation = line.split("=")[1]
        try:
            perturbation = float(perturbation)
        except ValueError:
            raise ValueError("Invalid file format")
        nonlocal i
        i += 1

    def get_variable_names(lines):
        nonlocal i
        nonlocal names_dict
        nonlocal index_dict
        for j in range(n):
            name = lines[i].strip()
            if name in names_dict:
                raise ValueError("Duplicate node name")
            if name in forbidden_names:
                raise ValueError("Invalid node name")
            for char in forbidden_chars:
                if char in name:
                    raise ValueError("Invalid node name")

            names_dict[name] = j
            index_dict[j] = name
            i += 1
        
    def checkline(line, expected):
        if line.strip() != expected:
            raise ValueError("Invalid file format")
        nonlocal i
        i += 1

    def get_function(line, probs):
        nonlocal nv
        nonlocal F
        nonlocal varFInt
        nonlocal i
        fun_expr = [x.strip() for x in lines[i].strip().split(":")]
        # get probability of the function
        try:
            fun_prob = float(fun_expr[0])
            probs.append(fun_prob)
        except ValueError:
            raise ValueError("Invalid file format")
        # extract variable names
        fun = fun_expr[1]
        vars = get_vars_from_assa_expr(fun)
        # check if extracted variables are valid
        unsorted_var_indices = []
        for var in vars:
            unsorted_var_indices.append(names_dict[var])
        updated_fun = fun
        for rep in logical_replacements:
            updated_fun = updated_fun.replace(rep, logical_replacements[rep])
        sorted_var_indices = sorted(unsorted_var_indices)
        nv.append(len(sorted_var_indices))

        # add variable indices to the list varFInt
        varFInt.append(sorted_var_indices)
        sorted_vars = [index_dict[var] for var in sorted_var_indices]

        # generate every possible variable evaluation
        truth_combinations = list(itertools.product([False, True], repeat=n))
        truth_table = []
        # evaluate the function for every possible combination of variables
        # and store the result in the truth table
        for combination in truth_combinations:
            values = dict(zip(sorted_vars, combination))
            evaluated = eval(updated_fun, {}, values)
            truth_table.append(evaluated)
        F.append(truth_table)
        i += 1
    
    def get_np_node(line):
        nonlocal np
        nonlocal i
        while lines[i].strip() != "endNpNode":
            node_name = lines[i].strip()
            if node_name not in names_dict:
                raise ValueError("Invalid file format")
            np.append(names_dict[node_name])
            i += 1
        np = sorted(np)
        np.append(n)
        i += 1

    

    with open(path, 'r') as f:
        # clean the file
        lines = delete_comments_and_empty_lines(f.readlines())
        
        # Read the type of the PBN
        get_type(lines[i])

        # Read the number of nodes
        get_n(lines[i])

        # Read the perturbation rate
        get_perturbation(lines[i])
        
        # Read the names of the nodes
        checkline(lines[i], "nodeNames")
        get_variable_names(lines)
        checkline(lines[i], "endNodeNames")
        
        # for every variable...
        for j in range(n):
            function_count = 0
            checkline(lines[i], "node " + index_dict[j])
            probs = []
            
            # ... extract the boolean functions for the variable
            while lines[i].strip() != "endNode":
                function_count += 1
                get_function(lines[i], probs)
            nf.append(function_count)
            cij.append(probs)
            i += 1

        assert len(F) == sum(nf)
        checkline(lines[i], "npNode")
        get_np_node(lines[i])
    model = PBN(n, nf, nv, F, varFInt, cij, perturbation, np)
    return model



def load_from_file(path, format='sbml'): 
    """
    Loads a PBN from files of format .pbn or .sbml.
    
    Parameters
    ----------
    path : str
        Path to the file of format .pbn.
    format : str, optional
        Choose the format. Can be either 'sbml' for files with .sbml format
        or 'assa' for files with .pbn format. Defaults to 'sbml'.

    Returns
    -------
    PBN
        PBN object representing the network from the file.
    """
    match format:
        case 'sbml':
            return load_sbml(path)
        case 'assa':
            return load_assa(path)
        case _:
            raise ValueError("Invalid format")
        







