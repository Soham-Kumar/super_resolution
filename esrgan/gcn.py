import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np

# Hyperparameters
no_of_edges = 700
gcn_in_channels = 3
gcn_hidden_channels = 128
gcn_out_channels = 64
gat_out_channels = 32
fused_embedding_dim = 4096
attention_dim = 64
num_nodes = 478
batch_size = 4

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels*8)
        self.conv4 = GCNConv(hidden_channels*8, hidden_channels*32)
        self.conv5 = GCNConv(hidden_channels*32, hidden_channels*64)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        # print(f"GCN1. {x.shape}")
        x = F.relu(self.conv2(x, edge_index))
        # print(f"2. {x.shape}")
        x = F.relu(self.conv3(x, edge_index))
        # print(f"3. {x.shape}")
        x = F.relu(self.conv4(x, edge_index))
        # print(f"4. {x.shape}")
        x = F.relu(self.conv5(x, edge_index))
        # print(f"5. {x.shape}")


        
        # Global pooling
        x = global_mean_pool(x, batch)
        # print(f"5. {x.shape}")

        return x

if __name__ == "__main__":
    def create_edge_index():
        feature_indices = {
            "Upperouterlip": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
            "Lowerouterlip": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
            "Upperinnerlip": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
            "Lowerinnerlip": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
            "NoseLeft": [8, 193, 245, 128, 114, 217, 209, 49, 48],
            "NoseRight": [278, 279, 429, 437, 343, 357, 465, 417, 8],
            "NoseBridge": [6, 197, 195, 5, 4, 1, 19],
            "EyeBrowsLeft": [107, 66, 105, 63, 70, 46, 53, 52, 65, 55, 107],
            "EyeBrowsRight": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
            "EyeLeftIn": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 246],
            "EyeLeftOut": [226, 247, 30, 29, 27, 28, 56, 190, 243, 112, 232, 231, 230, 229, 228, 31, 226],
            "EyeRightIn": [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362],
            "EyeRightOut": [464, 414, 286, 258, 257, 259, 260, 467, 446, 261, 448, 449, 450, 451, 452, 453, 464],
            "NoseCheekL": [114, 47, 100, 101, 50],
            "NoseEarL": [217, 126, 142, 36, 205, 187, 147, 93],
            "ForeheadEyebrowL": [9, 107, 66, 105, 63, 70, 156],
            "NoseCheekR": [343, 277, 329, 330, 280],
            "NoseEarR": [437, 355, 371, 266, 425, 411, 376, 323],
            "ForeheadEyebrowR": [9, 336, 296, 334, 293, 300, 383]
        }

        edges = []

        # Create edges between consecutive indices in each feature
        for feature, indices in feature_indices.items():
            for i in range(len(indices) - 1):
                edges.append((indices[i], indices[i + 1]))

        edges = np.array(edges)
        edge_vector = np.vstack((edges.flatten(), edges[:, ::-1].flatten()))
        edge_vector = torch.tensor(edge_vector, dtype=torch.long)

        return edge_vector

    edge_index = create_edge_index()
    
    # Create graph data for a single sample
    single_graph_data = torch.randn(478, 3)
    single_edge_index = edge_index

    # Create a batch of 4 samples
    batch_size = 4
    graph_data = torch.cat([single_graph_data for _ in range(batch_size)], dim=0)
    batch_edge_index = torch.cat([single_edge_index + (i * 478) for i in range(batch_size)], dim=1)

    # Create batch tensor
    batch_tensor = torch.repeat_interleave(torch.arange(batch_size), 478)

    # Move everything to device
    gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels).to(device)
    graph_data = graph_data.to(device)
    batch_edge_index = batch_edge_index.to(device)
    batch_tensor = batch_tensor.to(device)

    print(f"Graph Data Shape: {graph_data.shape}")
    print(f"Edge Index Shape: {batch_edge_index.shape}")
    print(f"Batch Tensor Shape: {batch_tensor.shape}")

    # Forward pass
    output = gcn(graph_data, batch_edge_index, batch_tensor)
    print(f"Output Shape: {output.shape}")

    # Print memory usage
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


    # total_params = sum(p.numel() for p in gcn.parameters()) / 1000000
    # print(f"Total number of parameters in the model: {total_params}")
    # for name, param in gcn.named_parameters():
    #     num_params = param.numel()
    #     num_params = num_params / 1000000
    #     print(f"Layer: {name}, Number of parameters: {num_params}")