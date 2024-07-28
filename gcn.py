import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool
from torch_geometric.data import Data, Batch

# Hyperparameters
no_of_edges = 700
gcn_in_channels = 3
gcn_hidden_channels = 16
gcn_out_channels = 64
gat_out_channels = 32
fused_embedding_dim = 512
attention_dim = 64
num_nodes = 478
batch_size = 4

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the GCN Model with GAT layer
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gat_out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.bn2 = BatchNorm(out_channels)
        self.conv3 = GCNConv(hidden_channels*2, out_channels)
        self.bn3 = BatchNorm(hidden_channels)
        
        # Add GAT layer
        self.gat = GATConv(hidden_channels, gat_out_channels, heads=4, concat=False)
        self.bn3 = BatchNorm(gat_out_channels)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(gat_out_channels, fused_embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        
        # Apply GAT layer
        x = self.gat(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        x = self.relu(x)
        return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F



# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         out = self.relu(self.conv1(x))
#         out = self.conv2(out)
#         out += residual
#         return self.relu(out)

# class SuperRes(nn.Module):
#     def __init__(self, num_residual_blocks, superres_in_channels, superres_hidden_channels, superres_res_channels, superres_out_channels):
#         super(SuperRes, self).__init__()
        
#         self.initial_conv = nn.Conv2d(superres_in_channels, superres_hidden_channels, kernel_size=3, padding=1)
        
#         self.residual_blocks = nn.Sequential(
#             *[ResidualBlock(superres_res_channels) for _ in range(num_residual_blocks)]
#         )
        
#         self.upscale1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(superres_hidden_channels, superres_hidden_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
        
#         self.upscale2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(superres_hidden_channels, superres_hidden_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
        
#         self.final_upscale = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(superres_hidden_channels, superres_out_channels, kernel_size=3, padding=1)
#         )

#     def forward(self, x):
#         x = self.initial_conv(x)
#         x = self.residual_blocks(x)
        
#         x1 = self.upscale1(x)  # 56x56
#         # print(f"SUPER RES : {x1.shape}")
#         # x2 = self.upscale2(x1)  # 112x112
#         # x3 = self.final_upscale(x2)  # 224x224
        
#         # return x1, x2, x3
#         return x1
