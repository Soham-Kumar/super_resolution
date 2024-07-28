import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class SuperRes(nn.Module):
    def __init__(self, num_residual_blocks, superres_in_channels, superres_hidden_channels, superres_res_channels, superres_out_channels):
        super(SuperRes, self).__init__()
        
        self.initial_conv = nn.Conv2d(superres_in_channels, superres_hidden_channels, kernel_size=3, padding=1)
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(superres_res_channels) for _ in range(num_residual_blocks)]
        )
        # self.residual_blocks_2 = nn.Sequential(
        #     *[ResidualBlock(superres_res_channels) for _ in range(num_residual_blocks)]
        # )
        # self.residual_blocks_3 = nn.Sequential(
        #     *[ResidualBlock(superres_res_channels) for _ in range(num_residual_blocks)]
        # )



        
        self.upscale1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(superres_hidden_channels, superres_hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upscale2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(superres_hidden_channels, superres_hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final_upscale = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(superres_hidden_channels, superres_out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        
        x1 = self.upscale1(x)  # Dimensions - 56x56
        # x1 = self.residual_blocks_2(x1)
        # print(f"SUPER RES : {x1.shape}")
        # x2 = self.upscale2(x1)  # 112x112
        # x3 = self.final_upscale(x2)  # 224x224
        
        # return x1, x2, x3
        return x1
# -------------------------------------------------------------------------------------------------------------
# # Example usage
# model = SuperRes()
# input_tensor = torch.randn(1, 3, 28, 28)  # [batch_size, channels, height, width]
# output = model(input_tensor)
# print(output.shape)  # Should print: torch.Size([1, 3, 224, 224])