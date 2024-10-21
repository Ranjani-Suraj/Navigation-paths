import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

A = [
    [0, 1,  0,  .8, 0],
    [0, 0,  .4, 0,  .3],
    [0, 0,  0,  0,  0],
    [0, 0,  .6, 0,  .7],
    [0, 0,  0,  .2, 0]]

G = nx.from_numpy_array(np.matrix(A), create_using=nx.DiGraph)
layout = nx.spring_layout(G)
nx.draw(G, layout)
nx.draw_networkx_edge_labels(G, pos=layout)
plt.show()