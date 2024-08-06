# -------------------------------------------------------------------------------------------------------------
# import numpy as np


# # Define facial feature indices
# feature_indices = {
#     "Upperouterlip": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
#     "Lowerouterlip": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
#     "Upperinnerlip": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
#     "Lowerinnerlip": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
#     "NoseLeft": [8, 193, 245, 128, 114, 217, 209, 49, 48],
#     "NoseRight": [278, 279, 429, 437, 343, 357, 465, 417, 8],
#     "NoseBridge": [6, 197, 195, 5, 4, 1, 19],
#     "EyeBrowsLeft": [107, 66, 105, 63, 70, 46, 53, 52, 65, 55, 107],
#     "EyeBrowsRight": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
#     "EyeLeftIn": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 246],
#     "EyeLeftOut": [226, 247, 30, 29, 27, 28, 56, 190, 243, 112, 232, 231, 230, 229, 228, 31, 226],
#     "EyeRightIn": [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362],
#     "EyeRightOut": [464, 414, 286, 258, 257, 259, 260, 467, 446, 261, 448, 449, 450, 451, 452, 453, 464],
#     "FaceOutline": [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
#                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162],
#     "NoseCheekL": [114, 47, 100, 101, 50],
#     "NoseEarL": [217, 126, 142, 36, 205, 187, 147, 93],
#     "ForeheadEyebrowL": [9, 107, 66, 105, 63, 70, 156],
#     "NoseCheekR": [343, 277, 329, 330, 280],
#     "NoseEarR": [437, 355, 371, 266, 425, 411, 376, 323],
#     "ForeheadEyebrowR": [9, 336, 296, 334, 293, 300, 383],
# }

# # Initialize an empty list to store the edges
# edges = []

# # Iterate through each feature and its corresponding indices
# for feature, indices in feature_indices.items():
#     # Create edges between each neighboring pair of points
#     for i in range(len(indices) - 1):
#         edges.append((indices[i], indices[i + 1]))

# # Convert the list of edges to a numpy array
# edges = np.array(edges)

# # Create the edge vector
# edge_vector = np.vstack((edges.flatten(), edges[:, ::-1].flatten()))

# print(edge_vector.shape)
# -------------------------------------------------------------------------------------------------------------


import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import pickle



if __name__ == "__main__":
    print("Starting...")
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
        

    dict = {}
    count = 0

    # Load dataset images
    # dataset_path = r'/home/sameenahmad/sameen/new_extracted_data/surprise'
    dataset_path = 'dataset'


    for file in os.listdir(dataset_path):
        if file.endswith(('.png')):
            image_path = os.path.join(dataset_path, file)
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, (64, 64))
        
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

            # Create MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect faces
            detection_result = face_landmarker.detect(mp_image)
            
            if detection_result.face_landmarks:
                landmarks, blendshapes, transform_matrix = convert_to_tensor(detection_result)
                dict[file] = landmarks
                count += 1
                if count % 1000 == 0:
                    print(f"{count} images processed")
                    

    filename = 'landmarks.pkl'             
    with open (filename, 'wb') as f:
        pickle.dump(dict, f)
    print("DONE")




