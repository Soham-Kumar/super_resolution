import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
# data loading
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# import mediapipe as mp # mediapipe
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# from facenet_pytorch import InceptionResnetV1 # face embeddings

import numpy as np
import pickle
from PIL import Image
import os

from gcn import GCN
from cmf import MSAF
from rrdb import RRDBNet
from esrgan_based import FinalModel




sr_size = 256
lr_size = 64

gcn_in_channels = 3
gcn_hidden_channels = 128
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
batch_size = 4
learning_rate = 0.001
num_epochs = 400

down_sample = 28

face_emb_reshape_dim = 32

msaf_in_channels = [8, 8]  # Number of input channels for MSAF
msaf_block_channel = 8 
msaf_block_dropout = 0.2 
msaf_reduction_factor = 4
msaf_split_block = 5


in_nc = 3
out_nc = 3
nf = 8
nb = 8
gc = 32




# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the trained model
model = FinalModel(gcn_in_channels, gcn_hidden_channels, gcn_out_channels, fused_embedding_dim,
                msaf_in_channels, msaf_block_channel, msaf_block_dropout, msaf_reduction_factor,
                in_nc, out_nc, nf, nb, gc)
model = model.to(device)
model.load_state_dict(torch.load('final_model_weights_esrgan_l1.pth', map_location=device))
model.eval()

# Image preprocessing
lr_transform = transforms.Compose([
    transforms.Resize((lr_size, lr_size)),
    transforms.ToTensor()
])




# --------------------------------- Edge_index
def create_edge_index(num_nodes):
    # Recreate edge_index logic here using num_nodes
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
    "FaceOutline": [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162],
    "NoseCheekL": [114, 47, 100, 101, 50],
    "NoseEarL": [217, 126, 142, 36, 205, 187, 147, 93],
    "ForeheadEyebrowL": [9, 107, 66, 105, 63, 70, 156],
    "NoseCheekR": [343, 277, 329, 330, 280],
    "NoseEarR": [437, 355, 371, 266, 425, 411, 376, 323],
    "ForeheadEyebrowR": [9, 336, 296, 334, 293, 300, 383],
    }
    edges = []
    for feature, indices in feature_indices.items():
        for i in range(len(indices) - 1):
            edges.append((indices[i], indices[i + 1]))
    edges = np.array(edges)
    edge_vector = np.vstack((edges.flatten(), edges[:, ::-1].flatten()))
    edge_vector = torch.tensor(edge_vector, dtype=torch.long)
    return edge_vector


# --------------------------------- Mediapipe 
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
        print("No face detected!")
        return None, None, None
    
    


def create_graph_data(num_nodes, image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (112, 112))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_face_small.jpg", image_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = face_landmarker.detect(mp_image)
    
    if detection_result.face_landmarks:
        landmarks, blendshapes, transform_matrix = convert_to_tensor(detection_result)
    else:
        print("No landmarks detected")
    x = landmarks
    edge_index = create_edge_index(num_nodes)
    return Data(x=x, edge_index=edge_index)

# Inference function
def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    lr_image = lr_transform(image).unsqueeze(0).to(device)
    graph_data = create_graph_data(478, image_path).to(device)
    
    with torch.no_grad():
        output = model(lr_image, graph_data)
    
    output_image = output.squeeze().cpu().permute(1, 2, 0).numpy()
    output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(output_image)

# Example usage
if __name__ == "__main__":
    # input_image_path = "test_face.jpg"
    # input_image_path = "test_0002_F_12_positive.jpg"
    input_image_path = "test_0053_M_30_positive.jpg"
    output_image = infer(input_image_path)
    output_image.save(f"output_image_{input_image_path.split(" ")[0]}.jpg")
    print("Inference completed. Output image saved as 'output_image.jpg'")
