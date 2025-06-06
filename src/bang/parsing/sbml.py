from libsbml import ASTNode, SBMLDocument, SBMLReader

from .bool_func import parseFunction


def enumerateNodes(qual_model):
    result: dict[str, int] = {}
    index = 0
    for node in qual_model.getListOfQualitativeSpecies():
        result[node.getName()] = index
        index += 1
    return result


# TODO: add errors
def parseSBMLDocument(path: str):
    reader = SBMLReader()
    doc: SBMLDocument = reader.readSBML(path)  # type: ignore
    F: list[list[bool]] = []
    nf: list[int]
    nv: list[int] = []
    varFInt: list[list[int]] = []

    model = doc.getModel()
    if model is None:
        pass

    qual_model = model.getPlugin("qual")
    if qual_model is None:
        pass

    nodes = enumerateNodes(qual_model)
    nf = [1 for n in nodes]

    for node_idx, transition in enumerate(
        qual_model.getListOfTransitions()
    ):  # Scan all the transitions.
        # Get the output variable
        # output = transition.getListOfOutputs()
        logic_terms = transition.getListOfFunctionTerms()
        if len(logic_terms) > 0:
            math: ASTNode = logic_terms[0].getMath()
            math.reduceToBinary()
            truth_table, relevant_nodes = parseFunction(math, nodes)
            F.append(truth_table)
            nv.append(len(relevant_nodes))
            varFInt.append(relevant_nodes)
        else:
            F.append([False, True])
            nv.append(1)
            varFInt.append([node_idx])

    return (
        len(nodes),
        nf,
        nv,
        F,
        varFInt,
        [[1.0] for node in nodes],
        0.0,
        [nodes[name] for name in nodes],
    )
