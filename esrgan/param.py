sr_size = 256
lr_size = 64

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

in_nc = 3
out_nc = 3
nf = 64
nb = 8
gc = 32


import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
import pickle
from PIL import Image
import os

from gcn import GCN
from cmf import MSAF
from rrdb import RRDBNet


def parameters_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels).to(device)
param_gcn = parameters_count(gcn)
print(f"GCN Model Parameters: {param_gcn}")
del gcn

msaf = MSAF(msaf_in_channels, msaf_block_channel, msaf_block_dropout, msaf_reduction_factor, msaf_split_block).to(device)
param_msaf = parameters_count(msaf)
print(f"MSAF Model Parameters: {param_msaf}")
del msaf

rrdb = RRDBNet(in_nc, out_nc, nf, nb, gc).to(device)
param_rrdb = parameters_count(rrdb)
print(f"RRDB Model Parameters: {param_rrdb}")
del rrdb

rechannel = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1).to(device)
param_channel = parameters_count(rechannel)
print(f"Rechannel Model Parameters: {param_channel}")
del rechannel


print(f"Total Parameters: {param_gcn + param_msaf + param_rrdb + param_channel}")