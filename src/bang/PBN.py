from typing import List
from libsbml import *

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
    
def skip_empty_lines(lines, i):
    while lines[i].strip() == "":
        i += 1
    return i

def load_sbml():
    pass

def load_assa(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        no_of_lines = len(lines)
       
        i = 0
        i = skip_empty_lines(lines, i)
        
        type_line = lines[i]
        type_line = type_line.strip()
        if type_line.startswith("type=") == False:
            raise ValueError("Invalid file format")
        type = type_line.split("=")[1]
        if type not in ['synchronous', 'rog', 'rmg', 'rmgrm', 'rmgro', 'rmgrorm', 'aro']
            raise ValueError("Invalid file format")
        
        i += 1
        i = skip_empty_lines(lines, i)
        
        n_line = lines[i]
        n_line = n_line.strip()
        if n_line.startswith("n=") == False:
            raise ValueError("Invalid file format")
        n = n_line.split("=")[1]
        if not n.isnumeric():
            raise ValueError("Invalid file format")
        n = int(n)

        i += 1
        i = skip_empty_lines(lines, i)

        perturbation_line = lines[i]
        perturbation_line = perturbation_line.strip()
        if perturbation_line.startswith("perturbation=") == False:
            raise ValueError("Invalid file format")
        perturbation = perturbation_line.split("=")[1]
        if not perturbation.isnumeric():
            raise ValueError("Invalid file format")
        perturbation = float(perturbation)

        i += 1
        pass
        # nf = list(map(int, lines[1].split()))
        # nv = list(map(int, lines[2].split()))
        # F = []
        # for i in range(sum(nf)):
        #     F.append(list(map(bool, lines[3 + i].split())))
        # varFInt = []
        # for i in range(n):
        #     varFInt.append(list(map(int, lines[3 + sum(nf) + i].split())))
        # cij = []
        # for i in range(n):
        #     cij.append(list(map(float, lines[3 + sum(nf) + n + i].split()))
        # perturbation = float(lines[3 + sum(nf) + 2*n])
        # npNode = list(map(int, lines[3 + sum(nf) + 2*n + 1].split()))
        # return PBN(n, nf, nv, F, varFInt, cij, perturbation, npNode)

def load_from_file(path, format='sbml'):
    match format:
        case 'sbml':
            load_sbml(path)
        case 'assa':
            load_assa(path)
        case _:
            raise ValueError("Invalid format")
