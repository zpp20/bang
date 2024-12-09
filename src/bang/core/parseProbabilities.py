from lxml import etree

def get_perturbation_rate(root :etree.ElementBase):
    pass

def get_function_probabilities(root :etree.ElementBase):
    pass

def get_exempt_nodes(root :etree.ElementBase):
    pass

def get_probabilities(path: str) -> tuple[dict[str, list[float]], float, list[str]]:
    root :etree.ElementBase = etree.parse(path, etree.XMLParser()).getroot()
    return {}, 0.0, []