from bang.core.PBN import PBN
import bang.parsing.assa 
import itertools



class Graph_PBN:
    def __init__(self, pbn):
        self.nodes = {i : PBN_Node(i, pbn, 0) for i in range(pbn.n)}
        self.pbn = pbn
        self.dfs_numbered = 0
        current_count = 0
        for i in range(pbn.n): # for every variable
            for j in range(current_count, current_count + pbn.nf[i]): # for every function of the variable
                for k in pbn.varFInt[j]: # for every variable that influences this function
                    self.nodes[i].in_nodes.append(k)
                    self.nodes[k].out_nodes.append(i)
            current_count += pbn.nf[i]
        for i in range(pbn.n):
            self.nodes[i].in_nodes = sorted(list(set(self.nodes[i].in_nodes))) # indices, not nodes! access nodes by self.nodes[i]
            self.nodes[i].out_nodes = sorted(list(set(self.nodes[i].out_nodes))) # indices, not nodes!
    

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
        for node in self.nodes.values():
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



    def find_scc(self):
        self.dfs_numerate() # assign dfs_id to each node
        for n in self.nodes.values():
            n.visited = False
        for node in self.nodes.values():
            if node.scc_id is None:
                for n in self.nodes.values():
                    n.visited = False
                self.scc_aux(node, node)


class PBN_Node:
    def __init__(self, id, pbn, current_value, name = "No name"):
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
graph.find_scc()
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
    

    







pbn2 = PBN(6, [1, 1, 1, 1, 1, 1], [2, 2, 1, 3, 2, 2], [[f1, f2, f3, f4, f5, f6]], [[0, 1], [0, 1], [1], [1,2,4], [3,4], [2,5]], [[1.], [1.], [1.], [1.], [1.], [1.]], 0.01, [2, 3, 4, 5])


graph2 = Graph_PBN(pbn2)
graph2.find_scc()
for i in graph2.nodes:
    print(graph2.nodes[i].scc_id)
       