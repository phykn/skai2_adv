import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from timm.models.layers import trunc_normal_


class ListLayerNorm(nn.Module):
    def __init__(
        self,
        channels: List[int]=[96, 192, 384, 768]
    ) -> None:
        super().__init__()
        self.norm_0 = nn.GroupNorm(1, channels[0], eps=1e-6)
        self.norm_1 = nn.GroupNorm(1, channels[1], eps=1e-6)
        self.norm_2 = nn.GroupNorm(1, channels[2], eps=1e-6)
        self.norm_3 = nn.GroupNorm(1, channels[3], eps=1e-6)

    def forward(
        self,
        xs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        xs[0] = self.norm_0(xs[0])
        xs[1] = self.norm_1(xs[1])
        xs[2] = self.norm_2(xs[2])
        xs[3] = self.norm_3(xs[3])
        return xs


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
        self.act_1 = nn.GELU()
        self.upscale_2 = UpScaleBlock(channels[1], channels[2], norm=True)
        self.act_2 = nn.GELU()
        self.upscale_3 = UpScaleBlock(channels[2], channels[3], norm=True)
        self.act_3 = nn.GELU()
        self.upscale_4 = UpScaleBlock(channels[3], channels[4], norm=True)
        self.act_4 = nn.GELU()
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
    
    
class GlobalAveragePooling2D(nn.Module):
    def __init__(
        self, 
        n_dim: int, 
        eps: float=1e-6
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_dim, eps=eps)
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.norm(x.mean([-2, -1]))

    
class ChangeKernel(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.conv(x)
