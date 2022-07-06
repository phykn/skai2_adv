import torch
import torch.nn as nn
from typing import List
from timm.models.layers import trunc_normal_


class UpScaleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool=True
    ) -> None:
        super().__init__()
        self.conv_layer = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1, 
            bias=False
        )
        self.norm_layer = nn.InstanceNorm2d(
            num_features=out_channels, 
            affine=True, 
            track_running_stats=True
        )
        self.norm = norm
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv_layer(x)
        if self.norm:
            x = self.norm_layer(x)
        return x

    
class ImageGenerator(nn.Module):
    def __init__(
        self,
        channels: List[int]=[768, 384, 192, 96, 64, 3]
    ) -> None:
        super().__init__()
        self.upscale_1 = UpScaleBlock(channels[0], channels[1], norm=True)
        self.act_1 = nn.SiLU()
        self.upscale_2 = UpScaleBlock(channels[1], channels[2], norm=True)
        self.act_2 = nn.SiLU()
        self.upscale_3 = UpScaleBlock(channels[2], channels[3], norm=True)
        self.act_3 = nn.SiLU()
        self.upscale_4 = UpScaleBlock(channels[3], channels[4], norm=True)
        self.act_4 = nn.SiLU()
        self.upscale_5 = UpScaleBlock(channels[4], channels[5], norm=False)        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(
        self,
        xs: List[torch.Tensor]
    ) -> torch.Tensor:
        x1 = self.act_1(self.upscale_1(xs[-1]))
        x2 = self.act_2(self.upscale_2(xs[-2]+x1))
        x3 = self.act_3(self.upscale_3(xs[-3]+x2))
        x4 = self.act_4(self.upscale_4(xs[-4]+x3))
        y = self.upscale_5(x4)
        return y