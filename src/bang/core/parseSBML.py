from libsbml import *
from .boolFunc import parseFunction
from .parseProbabilities import get_probabilities

def enumerateNodes(qual_model):
    result :dict[str, int] = {}
    index = 0
    for node in qual_model.getListOfQualitativeSpecies():
        result[node.getName()] = index
        index += 1
    return result

#TODO: add errors
def parseSBMLDocument(path: str):
    reader = SBMLReader()
    doc :SBMLDocument = reader.readSBML(path)
    F: list[list[bool]] = []
    nf :list[int]
    nv: list[int] = []
    varFInt: list[list[int]] = []
    func_prob, perturbation_rate, npNodes = get_probabilities(path)
    model = doc.getModel()
    if model is None:
        pass 

    qual_model = model.getPlugin("qual")
    if qual_model is None: 
        pass
        
    nodes = enumerateNodes(qual_model)
    nf = [0 for n in nodes]

    for transition in qual_model.getListOfTransitions():  # Scan all the transitions.
            # Get the output variable
            output :Output = transition.getListOfOutputs()[0]
            logic_terms = transition.getListOfFunctionTerms()
            if len(logic_terms) > 0:
                math :ASTNode = logic_terms[0].getMath()
                math.reduceToBinary()
                truth_table, relevant_nodes = parseFunction(math, nodes)
                F.append(truth_table)
                nv.append(len(relevant_nodes))
                varFInt.append(relevant_nodes)
                nf[nodes[output.getQualitativeSpecies()]] += 1
    npNode = [nodes[name] for name in npNodes]
    npNode.append(len(nodes))
    return len(nodes), nf, nv, F, varFInt, [func_prob[node] for node in nodes], perturbation_rate, npNode 
