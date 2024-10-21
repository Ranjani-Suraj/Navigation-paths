import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit_optimization.applications import Maxcut
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

'''n = 5  # Number of nodes in graph
G = nx.MultiDiGraph()
G.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 7.0), (1, 2, 1.0), (2, 3, 1.0), (4, 2, 1.0), (4, 1, 1.0), (4, 0, 1.0)]
# tuple is (i,j,weight) where (i,j) is the edge
for edge in elist:
    G.add_edge(edge[0], edge[1], weight = edge[2])

colors = ["r" for node in G.nodes()]
pos = nx.spring_layout(G)'''


def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()

def main_max_cut(G):
    nodes = list(G.nodes())
    w = nx.to_numpy_array(G)

    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()

    qubitOp, offset = qp.to_ising()

    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = exact.solve(qp)
    #print(result.fval)
    #print("Variable names: ", result.variable_names)
    out = []
    for i in result:
        out += [i]
    return out, nodes

#print(main_max_cut(G))

