from bang.core.PBN import PBN
#import bang.parsing.assa 
import itertools




class Graph_PBN:

    """ Representation of PBN as a graph, in order to perform graph algorithms on it.

    Attributes: 
    nodes: dict         A dictionary of PBN_Node objects, indexed by their id.
    pbn: PBN            The PBN object that is being represented as a graph. 
    dfs_numbered: int   The number of nodes that have been assigned a dfs_id so far.


    """
    def __init__(self, pbn):
        self.pbn = pbn
        self.nodes = {i : PBN_Node(i, pbn, 0) for i in range(pbn.n)}
        self.dfs_numbered = 0
        
        # assign in_nodes and out_nodes to each node
        current_count = 0
        for i in range(pbn.n): # for every variable
            for j in range(current_count, current_count + pbn.nf[i]): # for every function of the variable
                for k in pbn.varFInt[j]: # for every variable that influences this function
                    # we disregard in which function the variable is influencing the current function;
                    # we just need to know that it influences it
                    self.nodes[i].in_nodes.append(k)
                    self.nodes[k].out_nodes.append(i)
            current_count += pbn.nf[i]

        for i in range(pbn.n):
            self.nodes[i].in_nodes = sorted(list(set(self.nodes[i].in_nodes))) # indices, not nodes! access nodes by self.nodes[i]
            self.nodes[i].out_nodes = sorted(list(set(self.nodes[i].out_nodes))) # indices, not nodes!
        
        self.sccs = []
        self.blocks = []
        
    
# dfs numbering functions
    def dfs_aux(self, node):
        node.visited = True
        node.dfs_id = self.dfs_numbered
        self.dfs_numbered += 1
        for i in node.out_nodes:
            if not self.nodes[i].visited:
                self.dfs_aux(self.nodes[i])
        node.dfs_tree_size = self.dfs_numbered - node.dfs_id

    

    def dfs_numerate(self):
        self.dfs_numbered = 0
        for i in self.nodes:
            self.nodes[i].visited = False
        for node in self.nodes.values(): # nodes is dict index -> node
            if not node.visited:
                self.dfs_aux(node)

    
    def scc_aux(self, node, root_dfs):
        node.visited = True
        if node.scc_id is None:
            if node.dfs_id >= root_dfs.dfs_id and node.dfs_id < root_dfs.dfs_id + root_dfs.dfs_tree_size:
                node.scc_id = root_dfs.id
        for in_node in node.in_nodes:
            if not self.nodes[in_node].visited:
                self.scc_aux(self.nodes[in_node], root_dfs)



    def find_scc_and_blocks(self):
        self.dfs_numerate() # assign dfs_id to each node
        for n in self.nodes.values():
            n.visited = False
        for node in self.nodes.values():
            if node.scc_id is None:
                for n in self.nodes.values():
                    n.visited = False
                self.scc_aux(node, node)
        self.sccs = []
        sccs_ids = set([n.scc_id for n in self.nodes.values()])
        for scc_id in sccs_ids:
            self.sccs.append([n.id for n in self.nodes.values() if n.scc_id == scc_id])

        for scc in self.sccs:
            block = scc.copy()
            # influencers are nodes that are not in the block and influence at least one node in the block
            influencers = [node.id for node in self.nodes.values() if node.id not in block and any([i in block for i in node.out_nodes])]
            block += influencers
            block = sorted(list(set(block)))
            self.blocks.append(block)
        

        

        
        




class PBN_Node:
    """ Representation of a node in the PBN graph.

    Attributes:
    id: int  The id of the node.
    name: str The name of the node.
    current_value: int The current value of the node.
    in_nodes: list   A list of indices of nodes that influence this node.
    out_nodes: list  A list of indices of nodes that this node influences.
    functions: list A list of functions that update the value of the node.
    """
    def __init__(self, id, pbn, current_value, name = str(id)):
        self.id = id
        self.name = name
        self.current_value = current_value
        self.in_nodes = []
        self.out_nodes = []
        
        f_index = 0
        while f_index < id:
            f_index += pbn.nf[f_index]
        self.functions = pbn.F[f_index: f_index + pbn.nf[id]]
       
       # graph features
        self.dfs_id = None
        self.dfs_tree_size = None
        self.scc_id = None
        self.visited = False
    


        

pbn = PBN(2, [1, 1], [1, 1], [[True, False], [False, True]], [[1], [0]], [[1.], [1.]], 0.01, [2])

graph = Graph_PBN(pbn)
graph.dfs_numerate()
graph.find_scc_and_blocks()
# for i in graph.nodes:
#     print(graph.nodes[i].in_nodes)
#     print(graph.nodes[i].out_nodes)
#     print(graph.nodes[i].functions)
#     print(graph.nodes[i].dfs_id)
#     print(graph.nodes[i].dfs_tree_size)
#     print(graph.nodes[i].scc_id)

f1 = []
for i in range(48):
    f1.append(False)
for i in range(16):
    f1.append(True)
f2 = []
for i in range(32):
    f2.append(False)
for i in range(16):
    f2.append(True)
for i in range(16):
    f2.append(False)

f3 = []
for i in range(16):
    f1.append(True)
for i in range(16):
    f1.append(False)
for i in range(16):
    f1.append(True)
for i in range(16):
    f1.append(False)

f4 = []
for k in range(2):
    for i in range(3):
        for j in range(2):
            f4.append(False)
        for j in range(2):
            f4.append(True)
        for j in range(2):
            f4.append(False)
        for j in range(2):
            f4.append(True)
    for i in range(8):
        f4.append(True)

f5 = []

for i in range(8):
    for j in range(2):
        f5.append(False)
    for j in range(6):
        f5.append(True)

f6 = []

for i in range(2):
    for j in range(4):
        f6.append(False)
    for j in range(4):
        f6.append(True)
    for j in range(8):
        f6.append(False)
    for j in range(4):
        f6.append(False)
    for j in range(4):
        f6.append(True)
    for j in range(8):
        f6.append(False)
    
    
# This is the example from the paper of dr Mizera, page 10 on the left
pbn2 = PBN(6, [1, 1, 1, 1, 1, 1], [2, 2, 1, 3, 2, 2], [f1, f2, f3, f4, f5, f6], [[0, 1], [0, 1], [1], [1,2,4], [3,4], [2,5]], [[1.], [1.], [1.], [1.], [1.], [1.]], 0.01, [2, 3, 4, 5])


graph2 = Graph_PBN(pbn2)
graph2.find_scc_and_blocks()
for i in graph2.nodes:
    print(graph2.nodes[i].scc_id)
print(graph2.sccs)
print(graph2.blocks)


# Function to get the blocks from the paper of a PBN
# Input: PBN class object
# Output: list of blocks, where each block is a list of indices of nodes in the block
def get_blocks(pbn):
    graph = Graph_PBN(pbn)
    graph.find_scc_and_blocks()
    return graph.blocks