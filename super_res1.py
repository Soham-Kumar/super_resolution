import cv2
import mediapipe as mp
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from facenet_pytorch import InceptionResnetV1
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Hyperparameters
dir_path = r"lfw_funneled"
threshold = 0.4
batch_size = 8
gcn_in_channels = 3
gcn_hidden_channels = 32
gcn_out_channels = 64
hidden_dim = 128  # For attention fusion
# num_images_per_folder = 1
low_res_image_size = 112
high_res_image_size = 224
fused_embedding_dim = 1024
embedding_dim = 1024

losses = []

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Extract image embeddings from low-res images using pre-trained model
def extract_embeddings(hr_image_tensor, lr_image_tensor):
    # Image Embeddings (VGGFace)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    with torch.no_grad():
        image_embeddings = model(lr_image_tensor)
        print(f"Individual Image Embeddings: {image_embeddings.shape}")

    # Graph Embeddings (MediaPipe + GCN)
    base_options = mp_base_options.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = mp_face_landmarker.FaceLandmarkerOptions(base_options=base_options,
                                                       output_face_blendshapes=True,
                                                       output_facial_transformation_matrixes=True,
                                                       num_faces=1)
    detector = mp_face_landmarker.FaceLandmarker.create_from_options(options)
    image_embeddings_list = []
    graph_embeddings_list = []
    for i in range(hr_image_tensor.shape[0]):
        image_tensor = hr_image_tensor[i]
        numpy_array = image_tensor.cpu().numpy()
        if numpy_array.dtype == np.float32:
            numpy_array = (numpy_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(np.transpose(numpy_array, (1, 2, 0)))
        temp_image_path = f"temp_image_{i}.jpg"
        pil_image.save(temp_image_path)

        # Load image as MediaPipe Image object
        image = mp.Image.create_from_file(temp_image_path)

        # Detect landmarks and extract graph embeddings
        detection_result = detector.detect(image)
        # print(f"1. Detection Result: {len(detection_result.face_landmarks)} landmarks detected")
        if detection_result.face_landmarks:
            # print(f" Mediapipe Detection Result: {len(detection_result.face_landmarks[0])} landmarks detected")
            landmarks, blendshapes, transform_matrix = convert_to_tensor(detection_result)
            graph_data = create_graph_from_landmarks(landmarks, threshold)
            model = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)
            graph_embeddings = model(graph_data.x, graph_data.edge_index)

            # Debug prints
            # print(f"Individual Graph Embeddings after Mediapipe Landmarks: {graph_embeddings.shape}")

            # Append embeddings to lists
            image_embeddings_list.append(image_embeddings)
            graph_embeddings_list.append(graph_embeddings)
        else:
            print("No face detected!")
            # Show image using cv2
            # image = cv2.imread(temp_image_path)
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Delete the temporary file
        os.remove(temp_image_path)

    # Stack embeddings into tensors
    image_embeddings = torch.stack(image_embeddings_list)
    graph_embeddings = torch.stack(graph_embeddings_list)



    # Debug prints
    # print(f"Batch Image Embeddings: {image_embeddings.shape}")
    # print(f"Batch Graph Embeddings: {graph_embeddings.shape}")

    image_embeddings = torch.sum(image_embeddings, dim=1, keepdim=True)
    graph_embeddings = torch.sum(graph_embeddings, dim=1, keepdim=True)

    linear_layer_image = nn.Linear(512, embedding_dim).to(device)
    linear_layer_graph = nn.Linear(64, embedding_dim).to(device)

    image_embeddings = image_embeddings.to(device)
    graph_embeddings = graph_embeddings.to(device)

    image_embeddings = (linear_layer_image(image_embeddings)).squeeze()
    graph_embeddings = (linear_layer_graph(graph_embeddings)).squeeze()

    print(f"Image Embeddings: {image_embeddings.shape}")
    print(f"Graph Embeddings: {graph_embeddings.shape}")

    return image_embeddings, graph_embeddings


# Function to convert MediaPipe output to tensors
def convert_to_tensor(mediapipe_output):
    if (len(mediapipe_output.face_landmarks) > 0):
        landmarks = mediapipe_output.face_landmarks[0]
        landmarks = torch.tensor([[lm.x, lm.y, lm.z] for lm in landmarks])
        blendshapes = mediapipe_output.face_blendshapes[0]
        blendshapes = torch.tensor([cat.score for cat in blendshapes])
        transform_matrix = mediapipe_output.facial_transformation_matrixes[0]
        transform_matrix = torch.from_numpy(transform_matrix)
        return landmarks, blendshapes, transform_matrix
    else:
        print("No face detected!")
        
        return None, None, None


# Function to create graph from landmarks
def create_graph_from_landmarks(landmarks, threshold):
    num_nodes = len(landmarks)
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = torch.dist(landmarks[i], landmarks[j], p=1)
            if dist < threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=landmarks, edge_index=edge_index)
    return data


# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Attention Fusion Module
class AttentionFusionModule(nn.Module):
    def __init__(self, input_size):
        super(AttentionFusionModule, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size * 2, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.conv_reshape = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)


    def forward(self, embedding1, embedding2):
        # print(f"Image Embeddings (Attention Fusion): {embedding1.shape}")
        # print(f"Graph Embeddings (Attention Fusion): {embedding2.shape}")
        combined_embedding = torch.cat((embedding1, embedding2), dim=1)
        combined_embedding = F.tanh(self.linear1(combined_embedding))
        attention_weights = F.softmax(self.linear2(combined_embedding), dim=1)
        attended_embedding = torch.mul(attention_weights, embedding2)
        fused_embedding = embedding1 + attended_embedding
        # print(f"Fused Embeddings before reshaping (Attention Fusion): {fused_embedding.shape}")
        fused_embedding = fused_embedding.view(batch_size, 1, 32, 32)
        fused_embedding = self.conv_reshape(fused_embedding)
        print(f"Fused Embeddings: {fused_embedding.shape}")

        return fused_embedding



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv_transpose = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv_transpose(out)
        if out.shape != residual.shape:
            residual = self.conv_transpose(residual)
            out += residual
        else:
            out += residual
        out = self.relu(out)
        return out

class SuperResolution(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SuperResolution, self).__init__()
        
        self.residual_block1 = ResidualBlock(in_channels, out_channels)
        self.residual_block2 = ResidualBlock(out_channels, out_channels)
        self.residual_block3 = ResidualBlock(out_channels, out_channels)
        # Add one layer to resize the 256*256 image to 224*224
        self.reshape_layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=35, stride=1, padding=1)
        
    def forward(self, x):
        out = self.residual_block1(x)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.reshape_layer(out)
        print(f"Super Resolved Image: {out.shape}")
        return out



# Custom Dataset Class
class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        image_count = 0
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".jpg") and image_count <=2000:
                        img_path = os.path.join(folder_path, filename)
                        img = Image.open(img_path)
                        if img.size[0] >= high_res_image_size and img.size[1] >= high_res_image_size: 
                            self.image_paths.append(img_path)
                            image_count += 1


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Downsize and then upsize image to simulate low-res and high-res pair
        low_res_image = image.resize((low_res_image_size, low_res_image_size))
        high_res_image = image.resize((high_res_image_size, high_res_image_size))

        if self.transform:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)

        return low_res_image, high_res_image


# --- Training Loop ---
def train(super_resolution_model, attention_fusion, optimizer, data_loader, num_epochs):

    # Loss functions
    pixel_loss_fn = nn.L1Loss()
    perceptual_loss_fn = nn.MSELoss()

    super_resolution_model.to(device)
    attention_fusion.to(device) 

    # Perceptual model for feature extraction
    perceptual_model = InceptionResnetV1(pretrained='vggface2').eval()
    perceptual_model.to(device)

    for epoch in range(num_epochs):
        for i, (low_res_images, high_res_images) in enumerate(data_loader):
            low_res_images = low_res_images.to(device)
            high_res_images = high_res_images.to(device)

            # Forward pass
            image_embeddings, graph_embeddings = extract_embeddings(high_res_images, low_res_images)
            graph_embeddings = graph_embeddings.to(device)
            image_embeddings = image_embeddings.to(device)

            # print(f"Image Embeddings (training loop): {image_embeddings.shape}")
            # print(f"Graph Embeddings (training loop): {graph_embeddings.shape}")

            if(image_embeddings.shape[0] != 8):
                continue

            # Attention-based fusion 
            fused_embedding = attention_fusion(image_embeddings, graph_embeddings) 
            # print(f"Fused Embeddings (training loop): {fused_embedding.shape}")


            super_resolved_image = super_resolution_model(fused_embedding)
            # print(f"Super Resolved Image (training loop): {super_resolved_image.shape}")

            print(f"High Res Image Shape (training loop): {high_res_images.shape}")
            print(f"Super Resolved Image Shape (training loop): {super_resolved_image.shape}")
            # Pixel-wise loss
            pixel_loss = pixel_loss_fn(super_resolved_image, high_res_images)  # Use your high-res images

            # Perceptual loss
            sr_embeddings = perceptual_model(super_resolved_image)
            gt_embeddings = perceptual_model(high_res_images)
            perceptual_loss = perceptual_loss_fn(sr_embeddings, gt_embeddings)

            # Combine losses and backpropagate
            total_loss = pixel_loss + 0.1 * perceptual_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        
            print("----------------------------------------------------------------------------------")
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {total_loss.item():.4f}")
            print("----------------------------------------------------------------------------------")
            losses.append(total_loss.item())
        torch.save(super_resolution_model.state_dict(), f'super_res_weights.pt{epoch}')
        torch.save(attention_fusion.state_dict(), f'attention_fusion_weights.pt{epoch}')
        torch.save(graph_model.state_dict(), f'graph_model_weights.pt{epoch}')


# Data loading and model initialization
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = AffectNetDataset(dir_path, transform=transform) 
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
super_resolution_model = SuperResolution(out_channels=3)  
optimizer = torch.optim.Adam(super_resolution_model.parameters()) 
attention_fusion = AttentionFusionModule(embedding_dim) 
graph_model = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels)

# Move models to GPU before training
super_resolution_model.to(device)
attention_fusion.to(device)
graph_model.to(device)

# Train the model
train(super_resolution_model, attention_fusion, optimizer, data_loader, num_epochs=5)

# Save the weights
torch.save(super_resolution_model.state_dict(), 'super_res_weights.pt')
torch.save(attention_fusion.state_dict(), 'attention_fusion_weights.pt')
torch.save(graph_model.state_dict(), 'graph_model_weights.pt')


# Save the list of losses to a file
with open('losses.txt', 'w') as f:
    for loss in losses:
        f.write(str(loss) + '\n')
f.close()
