import torch
from torch.autograd import Variable
import torch.nn as nn
import GNN_Secondary
import numpy as np

dtypeFloat = torch.cuda.FloatTensor
dtypeLong = torch.cuda.LongTensor

node_count = 0

with open(f"C:/Users/samar/Downloads/graph_node_positions (1).txt", "r") as f:
    coords = f.readlines()[0].split(" ")
    for coord in coords:
        if coord != "output":
            node_count += 1
        else:
            break

num_nodes = int(node_count / 2)
num_neighbors = -1  # Could increase it!
train_filepath = f"C:/Users/samar/Downloads/old-tsp-data.tar/tsp-data/tsp-data/tsp{num_nodes}_train_concorde.txt"
hidden_dim = 300
num_layers = 5
mlp_layers = 2
learning_rate = 0.001
max_epochs = 60
batches_per_epoch = 10000

variables = {
    'train_filepath': f"C:/Users/samar/Downloads/old-tsp-data.tar/tsp-data/tsp-data/tsp{num_nodes}_train_concorde.txt",
    'val_filepath': f"C:/Users/samar/Downloads/old-tsp-data.tar/tsp-data/tsp-data/tsp{num_nodes}_val_concorde.txt",
    'test_filepath': f"C:/Users/samar/Downloads/graph_node_positions (1).txt",
    'num_nodes': num_nodes,
    'num_neighbors': num_neighbors,
    'node_dim': 2,
    'voc_nodes_in': 2,
    'voc_nodes_out': 2,
    'voc_edges_in': 3,
    'voc_edges_out': 2,
    'hidden_dim': hidden_dim,
    'num_layers': num_layers,
    'mlp_layers': mlp_layers,
    'aggregation': 'mean',
    'max_epochs': max_epochs,
    'val_every': 5,
    'test_every': 5,
    'batches_per_epoch': batches_per_epoch,
    'accumulation_steps': 1,
    'learning_rate': learning_rate,
    'decay_rate': 1.01
}

model = nn.DataParallel(
    GNN_Secondary.ResidualGatedGCNModel(variables, torch.cuda.FloatTensor, torch.cuda.LongTensor))
model.load_state_dict(torch.load("Models/GraphML_GNN.pth"))
model.eval()

num_samples = 2
num_nodes = variables['num_nodes']
num_neighbors = variables['num_neighbors']
test_filepath = variables['test_filepath']

dataset = iter(GNN_Secondary.GoogleTSPReader(num_nodes, num_neighbors, 1, test_filepath))
x_edges = []
x_edges_values = []
x_nodes = []
x_nodes_coord = []
y_edges = []
y_nodes = []
y_preds = []

with torch.no_grad():
    sample = next(dataset)
    for i in range(num_samples):
        # Convert batch to torch Variables
        x_edges.append(Variable(torch.LongTensor(sample.edges).type(dtypeLong), requires_grad=False))
        x_edges_values.append(Variable(torch.FloatTensor(sample.edges_values).type(dtypeFloat), requires_grad=False))
        x_nodes.append(Variable(torch.LongTensor(sample.nodes).type(dtypeLong), requires_grad=False))
        x_nodes_coord.append(Variable(torch.FloatTensor(sample.nodes_coord).type(dtypeFloat), requires_grad=False))
        y_edges.append(Variable(torch.LongTensor(sample.edges_target).type(dtypeLong), requires_grad=False))
        y_nodes.append(Variable(torch.LongTensor(sample.nodes_target).type(dtypeLong), requires_grad=False))

        # Compute class weights
        edge_labels = (y_edges[-1].cpu().numpy().flatten())
        edge_cw = GNN_Secondary.compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

        # Forward pass
        y_pred, loss = model.forward(x_edges[-1], x_edges_values[-1], x_nodes[-1], x_nodes_coord[-1], y_edges[-1],
                                     edge_cw)
        y_preds.append(y_pred)

y_preds = torch.squeeze(torch.stack(y_preds))

GNN_Secondary.plot_predictions(x_nodes_coord, x_edges, x_edges_values, y_edges, y_preds, num_plots=num_samples)
