import torch
from torchviz import make_dot

from esrgan_based import FinalModel
from torch_geometric.data import Data
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model parameters
gcn_in_channels = 3
gcn_hidden_channels = 128
gcn_out_channels = 32
fused_embedding_dim = 512
msaf_in_channels = [8, 8]
msaf_block_channel = 8
msaf_block_dropout = 0.2
msaf_reduction_factor = 4
in_nc = 3
out_nc = 3
nf = 8
nb = 8
gc = 32

# Initialize the model
model = FinalModel(gcn_in_channels, gcn_hidden_channels, gcn_out_channels, fused_embedding_dim,
                   msaf_in_channels, msaf_block_channel, msaf_block_dropout, msaf_reduction_factor,
                   in_nc, out_nc, nf, nb, gc)
model = model.to(device)

# Generate random inputs
lr_image = torch.randn(4, 3, 64, 64).to(device)  # Low-resolution image
graph_x = torch.randn(4 * 478, 3).to(device)     # Node features (batch of graphs with 478 nodes each)
edge_index = torch.randint(0, 478, (2, 700 * 4)).to(device)  # Random edges (700 edges per graph)
batch = torch.cat([torch.full((478,), i) for i in range(4)]).to(device)  # Batch indices

graph_data = Data(x=graph_x, edge_index=edge_index, batch=batch)

# Forward pass
outputs = model(lr_image, graph_data)

# Visualize the model
graph = make_dot(outputs, params=dict(list(model.named_parameters()) + [('lr_image', lr_image), ('graph_x', graph_x), ('edge_index', edge_index)]))
graph.render("FinalModel", format="png")
