import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.GroupNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(
        self,
        xs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        xs[0] = self.norm_0(xs[0])
        xs[1] = self.norm_1(xs[1])
        xs[2] = self.norm_2(xs[2])
        xs[3] = self.norm_3(xs[3])
        return xs


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


class PositionEmbeddingSine(nn.Module):
    """
    source: https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    Modified: Delete about mask
    """
    def __init__(
        self, 
        num_pos_feats: int=64, 
        temperature: int=10000, 
        normalize: bool=False, 
        scale: Optional[float]=None
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        N, C, H, W = x.shape
        mask = torch.full((N, H, W), False, dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.floor(dim_t / 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    
class FeedForwardNetwork(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int=512, 
        dropout=0.1
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

    
class AttentionEncodeLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        d_ff: int=512, 
        dropout: float=0.1
    ) -> None:
        super().__init__()
        self.att_layer = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff=d_ff, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.apply(self._init_weights)

    @staticmethod
    def bcwh2nbc(
        x: torch.Tensor
    ) -> torch.Tensor:
        return x.flatten(2).permute(2, 0, 1)

    @staticmethod
    def nbc2bcwh(
        x: torch.Tensor
    ) -> torch.Tensor:
        n, b, c = x.shape
        w = h = math.sqrt(n)
        return x.permute(1, 2, 0).view(b, c, int(w), int(h))
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.bcwh2nbc(x)
        p = self.bcwh2nbc(p)

        q = k = x + p
        y, score = self.att_layer(q, k, x)
        x = self.norm_1(x+self.dropout_1(y))
        x = self.norm_2(x+self.dropout_2(self.ffn(x)))
        y = self.nbc2bcwh(x)
        return y, score
    
    
class GlobalAveragePooling2D(nn.Module):
    def __init__(
        self, 
        n_dim: int, 
        eps: float=1e-6
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_dim, eps=eps)
        self.norm.weight.data.fill_(1.0)
        self.norm.bias.data.zero_() 
        
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