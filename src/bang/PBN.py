from typing import List

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
    


def load_sbml():
    pass

def load_assa():
    pass

def load_from_file(format='sbml'):
    match format:
        case 'sbml':
            load_sbml()
        case 'assa':
            load_assa()
        case _:
            raise ValueError("Invalid format")
