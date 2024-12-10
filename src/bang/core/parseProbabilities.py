from lxml import etree

qual_namespace ="{http://www.sbml.org/sbml/level3/version1/qual/version1}"

def get_perturbation_rate(model :etree.ElementBase):
    pert_element :etree.ElementBase = model.find("{*}perturbation", None)
    if pert_element is None:
        return 0.0
    return float(pert_element.attrib["value"])

def get_function_probabilities(model :etree.ElementBase):
    func_list :list[etree.ElementBase] = model.find("{*}listOfTransitions", None).getchildren()
    result :dict[str, list[float]] = {}
    for func in func_list:
        name = func.find("{*}listOfOutputs", None).find("{*}output").attrib[qual_namespace + "qualitativeSpecies"]
        if not name in result:
            result[name] = [float(func.attrib[qual_namespace + "probability"]) if qual_namespace + "probability" in func.attrib else 1.0]
        else:
            result[name].append(float(func.attrib[qual_namespace + "probability"]))
    return result
            

def get_exempt_nodes(model :etree.ElementBase):
    node_list :list[etree.ElementBase] = model.find("{*}listOfQualitativeSpecies", None).getchildren()
    result = []
    for node in node_list:
        if "exempt" in node.attrib and node.attrib[qual_namespace + "exempt"] == True:
            result.append(node.attrib["name"])
    return result

def get_probabilities(path: str) -> tuple[dict[str, list[float]], float, list[str]]:
    root :etree.ElementBase = etree.parse(path, etree.XMLParser()).getroot()
    model :etree.ElementBase = root.getchildren()[0]
    return get_function_probabilities(model), get_perturbation_rate(model), get_exempt_nodes(model)