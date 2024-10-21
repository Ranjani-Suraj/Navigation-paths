from itertools import permutations
import matplotlib.pyplot as plt
import networkx as nx
from qiskit_optimization.applications import Tsp


def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()

def draw_tsp_solution(G, order, colors, pos):
    G2 = nx.DiGraph()
    G2.add_nodes_from(G)
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)
    plt.show()

def brute_force_tsp(w, N):
    mapping = {}
    for i in range(len(N)):
        mapping[i] = N[i]
    #print("mapping is: ", mapping)

    a = list(permutations(range(1, len(N))))
    last_best_distance = 1e10
    for i in a:
        distance = 0
        pre_j = 0
        for j in i:
            distance = distance + w[j, pre_j]
            pre_j = j
        distance = distance + w[pre_j, 0]
        order = (0,) + i
        if distance < last_best_distance:
            best_order = order
            last_best_distance = distance
            #print("order = " + str(order) + " Distance = " + str(distance))
    
    best_order = list(best_order)
    print("best order is: ", best_order)
    for j in range(len(best_order)):
        best_order[j] = mapping[best_order[j]]
    
    return last_best_distance, best_order

'''n = 5
vertices = [0, 1, 2, 3, 4]
edges = [(0, 1), (0, 3), (0, 4), (1, 2), (2, 3), (4, 1)]
G = nx.Graph()
G.add_nodes_from(vertices)
for edge in edges:
    G.add_edge(edge[0], edge[1], weight = 3)'''

def main_tsp(G):
    #tsp = Tsp.create_random_instance(n, seed=123)
    tsp = Tsp(G)
    adj_matrix = nx.to_numpy_array(tsp.graph)
    #print("distance\n", adj_matrix)

    colors = ["r" for node in tsp.graph.nodes]
    pos=nx.spring_layout(G)
    #draw_graph(tsp.graph, colors, pos)


    best_distance, best_order = brute_force_tsp(adj_matrix, list(G.nodes()))
    '''print(
        "Best order from brute force = "
        + str(best_order)
        + " with total distance = "
        + str(best_distance)
    )'''
    return best_order



'''G = nx.Graph()
G.add_nodes_from(vertices)
for edge in edges:
    G.add_edge(edge[0], edge[1])


qp = tsp.to_quadratic_program()

qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
qubitOp, offset = qubo.to_ising()

print("marker 1")

exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result = exact.solve(qubo)

print("marker 2")

ee = NumPyMinimumEigensolver()
result = ee.compute_minimum_eigenvalue(qubitOp)
x = tsp.sample_most_likely(result.eigenstate)

print("marker 3")

z = tsp.interpret(x)
print("solution:", z)
draw_tsp_solution(tsp.graph, z, colors, pos)
'''