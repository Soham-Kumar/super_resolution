# FaceNet PyTorch for Facial Embeddings
# Mediapipe for Facial Landmarks
# GCN for Graph embeddings from Facial Landmarks
# Fusion with cmf.py [understand the model]
# Super-Resolution with basic deconv+conv layers


# 1. Create Edge Vector (with base node at centre of nose -1)
# 2. Load Image and Graph Data
# 3. Mediapipe Functions
# 
# 



# Sizes
# Super-Resolution: 224x224 from 112x112
# Facial Landmarks: 478, Edges: 700
# Face Embeddings: 512, Graph Embeddings: 512
# Fused Embeddings: 512 -> [8,8,8]


import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
# data loading
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

import mediapipe as mp # mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from facenet_pytorch import InceptionResnetV1 # face embeddings

import numpy as np
import pickle
from PIL import Image
import os

from rdn import RDN
from gcn import GCN
from cmf import MSAF
from super_res import SuperRes


sr_size = 224
lr_size = 112

gcn_in_channels = 3
gcn_hidden_channels = 16
gcn_res_channels = 32
num_residual_blocks = 8
gcn_out_channels = 32
gat_out_channels = 32
fused_embedding_dim = 512

superres_in_channels = 3
superres_hidden_channels = 32
superres_res_channels = 32
superres_out_channels = 3

attention_dim = 64
num_nodes = 478
batch_size = 32
learning_rate = 0.001
num_epochs = 400

down_sample = 28

face_emb_reshape_dim = 8

msaf_in_channels = [8, 8]  # Number of input channels for MSAF
msaf_block_channel = 8 
msaf_block_dropout = 0.2 
msaf_reduction_factor = 4
msaf_split_block = 5

msaf1_in_channels = [8, 8]



# Paths Server
# image_path = '/home/sameenahmad/sameen/new_extracted_data/surprise'
# graph_path = 'landmarks_balanced.pkl'
# Paths Local
image_path = 'lfw'
graph_path = 'landmarks_balanced.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# STEP 1: Create FaceDetector object.
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
detector_options = vision.FaceDetectorOptions(base_options=base_options)
face_detector = vision.FaceDetector.create_from_options(detector_options)

# STEP 2: Create FaceLandmarker object.
landmarker_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
landmarker_options = vision.FaceLandmarkerOptions(base_options=landmarker_options,
                                                  output_face_blendshapes=True,
                                                  output_facial_transformation_matrixes=True,
                                                  num_faces=1)
face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)


def convert_to_tensor(mediapipe_output):
    if len(mediapipe_output.face_landmarks) > 0:
        landmarks = mediapipe_output.face_landmarks[0]
        landmarks = torch.tensor([[lm.x, lm.y, lm.z] for lm in landmarks])
        blendshapes = mediapipe_output.face_blendshapes[0]
        blendshapes = torch.tensor([cat.score for cat in blendshapes])
        transform_matrix = mediapipe_output.facial_transformation_matrixes[0]
        transform_matrix = torch.from_numpy(transform_matrix)
        return landmarks, blendshapes, transform_matrix
    else:
        return None, None, None


class FinalModel(nn.Module):
    def __init__(self,
                 gcn_in_channels, gcn_hidden_channels, gcn_out_channels, fused_embedding_dim,
                 msaf_in_channels, msaf_block_channel, msaf_block_dropout, msaf_reduction_factor,
                 superres_in_channels, superres_hidden_channels, superres_res_channels, superres_out_channels):
        super(FinalModel, self).__init__()
        self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels, gat_out_channels)
        self.face_net = InceptionResnetV1(pretrained='vggface2')
        
        # Initial fusion of face and graph embeddings
        self.msaf_initial = MSAF(msaf_in_channels, msaf_block_channel, msaf_block_dropout, reduction_factor=msaf_reduction_factor, split_block=msaf_split_block)
        
        # Fusion with intermediate SuperRes steps
        self.msaf_56 = MSAF([superres_hidden_channels, superres_hidden_channels], msaf_block_channel, msaf_block_dropout, reduction_factor=msaf_reduction_factor, split_block=msaf_split_block)
        self.msaf_112 = MSAF([superres_hidden_channels, superres_hidden_channels], msaf_block_channel, msaf_block_dropout, reduction_factor=msaf_reduction_factor, split_block=msaf_split_block)
        
        self.SuperRes = SuperRes(num_residual_blocks=8,
                                 superres_in_channels=superres_in_channels,
                                 superres_hidden_channels=superres_hidden_channels,
                                 superres_res_channels=superres_res_channels,
                                 superres_out_channels=superres_out_channels,)
                
        self.fc = nn.Linear(msaf_in_channels[0]*face_emb_reshape_dim*face_emb_reshape_dim*2, 3*down_sample*down_sample)
        
        
        # Upscale layers
        self.upscale_56 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(3, superres_hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.rechannel_conv = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        # self.upscale_112 = nn.Sequential(
        #     nn.Upsample(scale_factor=4, mode='nearest'),
        #     nn.Conv2d(3, superres_hidden_channels, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, lr_image, graph_data):
        
        # Face embedding from the image
        face_embedding = self.face_net(lr_image)
        face_embedding = face_embedding.view(-1, msaf_in_channels[0], face_emb_reshape_dim, face_emb_reshape_dim)

        # Graph embedding
        graph_embedding = self.gcn(graph_data.x, graph_data.edge_index, graph_data.batch)
        graph_embedding = graph_embedding.view(-1, msaf_in_channels[1], face_emb_reshape_dim, face_emb_reshape_dim)

        # Initial fusion of face and graph embeddings
        fused_embeddings = self.msaf_initial([face_embedding, graph_embedding])
        combined_embedding = torch.cat(fused_embeddings, dim=1)
        combined_embedding = combined_embedding.view(combined_embedding.size(0), -1)
        combined_embedding = self.fc(combined_embedding)
        combined_embedding = combined_embedding.view(-1, 3, down_sample, down_sample)

        # Super Resolution intermediate steps
        sr_56 = self.SuperRes(combined_embedding)

        # print(f"SR 56 Shape: {sr_56.shape}")

        # Upscale the [batch,8,8,8] fused embedding to 56x56 for merging
        fused_56 = self.upscale_56(combined_embedding)

        # Fusion of combined embeddings and intermediate SuperRes output
        fused_56 = self.msaf_56([sr_56, fused_56])
        fused_56 = torch.cat(fused_56, dim=1) # [batch, 64, 56, 56]

        fused_56 = self.rechannel_conv(fused_56) # [batch, 32, 56, 56]

        fused_112 = self.SuperRes.upscale2(fused_56) # [batch, 32, 112, 112]

        # Final upscaling
        final_output = self.SuperRes.final_upscale(fused_112) # [batch, 3, 224, 224]
        return final_output
    

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, image_path, graph_path, lr_transform=None, hr_transform=None):
        self.image_path = image_path
        self.graph_path = graph_path
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.graph_data = self.load_graph_data()

    def load_graph_data(self):
        with open(self.graph_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        image_name = list(self.graph_data.keys())[idx]
        image = Image.open(f"{self.image_path}/{image_name}").convert('RGB')
        if self.lr_transform:
            lr_image = self.lr_transform(image)
        if self.hr_transform:
            hr_image = self.hr_transform(image)

        image_path = f"{self.image_path}/{image_name}"
        graph_x = self.graph_data[image_name]
        edge_index = self.create_edge_index(graph_x.shape[0])

        data = Data(x=graph_x, edge_index=edge_index)
        return lr_image, hr_image, image_path, data

    def create_edge_index(self, num_nodes):
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


        # print(f"Edge index shape: {edge_vector.shape}")
        # print(f"Number of unique edges: {edge_vector.shape[1] // 2}")
        # print(f"Total number of edges (including both directions): {edge_vector.shape[1]}")
        
        return edge_vector




    
if __name__ == '__main__':

    # Image preprocessing
    lr_transform = transforms.Compose([
        transforms.Resize((lr_size, lr_size)),
        transforms.ToTensor()
    ])
    hr_transform = transforms.Compose([
        transforms.Resize((sr_size, sr_size)),
        transforms.ToTensor()
    ])

    # Initialize dataset and dataloader
    dataset = CustomDataset(image_path, graph_path, lr_transform=lr_transform, hr_transform=hr_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = FinalModel(gcn_in_channels, gcn_hidden_channels, gcn_out_channels, fused_embedding_dim,
                 msaf_in_channels, msaf_block_channel, msaf_block_dropout, msaf_reduction_factor,
                 superres_in_channels, superres_hidden_channels, superres_res_channels, superres_out_channels)
    
    model = model.to(device)


    perceptual_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


    def count_parameters(model):
        total_params = 0
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name}: {module_params:,} parameters")
            total_params += module_params
        print(f"Total: {total_params:,} parameters")
        return total_params
    
    total_params = count_parameters(model)

    # Define loss function and optimizer
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    loss_file_path = "training_loss.txt"
    if not os.path.exists(loss_file_path):
        open(loss_file_path, 'w').close()
    loss_file = open(loss_file_path, "a")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (lr_image, hr_image, image_path, graph_data) in enumerate(dataloader):
            lr_image, hr_image, graph_data = lr_image.to(device), hr_image.to(device), graph_data.to(device)
            landmark_array = []
            optimizer.zero_grad()
            outputs = model(lr_image, graph_data)

            # Compute loss and backpropagate
            L1_loss = criterion(outputs, hr_image)
            perceptual_loss = criterion(perceptual_model(outputs), perceptual_model(hr_image))
            # print(f"L1 Loss: {L1_loss.item()}, Perceptual Loss: {10*perceptual_loss.item()}")

            loss = L1_loss + 5*perceptual_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
       
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
        loss_file.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}\n")

    print("Training completed.")

    # Save the model weights
    torch.save(model.state_dict(), 'final_model_weights_perceptual.pth')
    print("Model weights saved.")
    loss_file.close()



























# -------------------------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool
# from torch_geometric.data import Data, Batch

# # Hyperparameters
# no_of_edges = 700
# gcn_in_channels = 3
# gcn_hidden_channels = 16
# gcn_out_channels = 64
# gat_out_channels = 32
# fused_embedding_dim = 512
# attention_dim = 64
# num_nodes = 478
# batch_size = 4

# # Check for GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Define the GCN Model with GAT layer
# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, gat_out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.bn1 = BatchNorm(hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
#         self.bn2 = BatchNorm(out_channels)
#         self.conv3 = GCNConv(hidden_channels*2, out_channels)
#         self.bn3 = BatchNorm(hidden_channels)
        
#         # Add GAT layer
#         self.gat = GATConv(hidden_channels, gat_out_channels, heads=4, concat=False)
#         self.bn3 = BatchNorm(gat_out_channels)
        
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(gat_out_channels, fused_embedding_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = F.relu(x)

        
#         # Apply GAT layer
#         x = self.gat(x, edge_index)
#         x = self.bn3(x)
#         x = F.relu(x)
        
#         # Global pooling
#         x = global_mean_pool(x, batch)
#         x = self.fc(x)
#         x = self.relu(x)
#         return x


# -------------------------------------------------------------------------------------------------------------
# super_res.py
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
#     def __init__(self, num_residual_blocks=8):
#         super(SuperRes, self).__init__()
        
#         self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
#         self.residual_blocks = nn.Sequential(
#             *[ResidualBlock(64) for _ in range(num_residual_blocks)]
#         )
        
#         self.upscale = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(64, 3, kernel_size=3, padding=1)
#         )

#     def forward(self, x):
#         print("INSIDE SUPER RES")
#         x = self.initial_conv(x)
#         # print(f"Initial Conv Shape: {x.shape}")
#         x = self.residual_blocks(x)
#         # print(f"Residual Blocks Shape: {x.shape}")
#         x = self.upscale(x)
#         # print(f"Upscale Shape: {x.shape}")
#         # print("SUPER RES END")
#         return x