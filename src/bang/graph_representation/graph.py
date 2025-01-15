from bang.core.PBN import PBN
import bang.parsing.assa 
import itertools



class Graph_PBN:
    def __init__(self, pbn):
        self.nodes = {i : PBN_Node(i, pbn, 0) for i in range(pbn.n)}
        self.pbn = pbn

        current_count = 0
        for i in range(pbn.n): # for every variable
            for j in range(current_count, current_count + pbn.nf[i]): # for every function of the variable
                for k in pbn.varFInt[j]: # for every variable that influences this function
                    self.nodes[i].in_nodes.append(k)
                    self.nodes[k].out_nodes.append(i)
            current_count += pbn.nf[i]
        for i in range(pbn.n):
            self.nodes[i].in_nodes = sorted(list(set(self.nodes[i].in_nodes)))
            self.nodes[i].out_nodes = sorted(list(set(self.nodes[i].out_nodes)))
                    

        


class PBN_Node:
    def __init__(self, id, pbn, current_value, name = "No name"):
        self.id = id
        assert id < len(pbn.F), "Node id out of range"
        assert id >= 0, "Node id out of range"
        self.name = name
        self.current_value = current_value
        self.in_nodes = []
        self.out_nodes = []
        f_index = 0
        while f_index < id:
            f_index += pbn.nf[f_index]
        self.functions = pbn.F[f_index: f_index + pbn.nf[id]]
        self.dfs_id = None
        self.dfs_subtree_size = None
        self.scc_id = None


        

pbn = PBN(2, [1, 1], [1, 1], [[True, False], [False, True]], [[1], [0]], [[1.], [1.]], 0.01, [2])

graph = Graph_PBN(pbn)
for i in graph.nodes:
    print(graph.nodes[i].in_nodes)
    print(graph.nodes[i].out_nodes)
    print(graph.nodes[i].functions)
    print("\n")



       