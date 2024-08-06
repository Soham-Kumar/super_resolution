import torch
from torch import nn
from rrdb import RRDBNet

# Assuming the RRDBNet class is already defined as in your previous code

# Create an instance of RRDBNet
# Parameters: in_nc, out_nc, nf, nb, gc
model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)

# Create a random input tensor
# Let's assume we're working with a 64x64 RGB image
batch_size = 1
channels = 3
height = 64
width = 64

input_tensor = torch.randn(batch_size, channels, height, width)

# Print input shape
print(f"Input shape: {input_tensor.shape}")

# Run the input through the model
with torch.no_grad():
    output = model(input_tensor)

# Print output shape
print(f"Output shape: {output.shape}")