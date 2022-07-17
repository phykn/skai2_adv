import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from timm.loss import LabelSmoothingCrossEntropy
from .convnext import convnext_tiny
from .layers import ListLayerNorm, ChangeKernel
from .transformer import PositionEmbedding1D, PositionEmbedding2D, Transformer


class Classifier(nn.Module):
    def __init__(
        self,
        num_class: int=5,
        pretrained: bool=True,
        drop_path_rate: float=0.1,
        dropout: float=0.1,
        d_model: int=256,
        n_head: int=8, 
        num_encoder_layers: int=6,
        num_decoder_layers: int=6, 
        d_ff: int=1024,
        label_smoothing: float=0.1
    ) -> None:
        super().__init__()
        num_kernels = [96, 192, 384, 768]
        self.convnext = convnext_tiny(
            pretrained=True, 
            drop_path_rate=drop_path_rate
        )        
        self.norm = ListLayerNorm(num_kernels)
        self.change = ChangeKernel(
            in_channels=num_kernels[-1], 
            out_channels=d_model
        )
        
        self.pe_1d = PositionEmbedding1D()        
        self.pe_2d = PositionEmbedding2D(
            num_pos_feats=d_model//2,
            normalize=True
        )
        self.embed = nn.Embedding(
            num_embeddings=num_class, 
            embedding_dim=d_model
        )
        nn.init.xavier_uniform_(self.embed.weight)
        
        self.transformer = Transformer(
            d_model=d_model,
            n_head=n_head, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, 
            d_ff=d_ff, 
            dropout=dropout
        )
        
        self.classifier = nn.Linear(
            in_features=d_model,
            out_features=2
        )
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
        self.ce_loss = LabelSmoothingCrossEntropy(label_smoothing)
        
    @staticmethod
    def bcwh2nbc(
        x: torch.Tensor
    ) -> torch.Tensor:
        return x.flatten(2).permute(2, 0, 1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        # convnext
        xs = self.convnext(x)
        xs = self.norm(xs)
        x = xs[-1]
        
        # x, p
        x = self.change(x)
        p = self.pe_2d(x)
        
        x = self.bcwh2nbc(x)
        p = self.bcwh2nbc(p)
        
        # add noise
        if self.training:
            x = x + torch.randn_like(x)
            
        # y, q
        y = self.embed(y).permute(1, 0, 2)
        q = self.pe_1d(y)
        
        # transformer
        o = self.transformer(x, p, y, q)
        o = self.classifier(o)
        return o
    
    def loss(
        self,
        data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = data["image"]
        y = data["query"]
        t = data["label"]
        
        o = self(x, y).flatten(0, 1)
        t = t.flatten(0)        
        loss = self.ce_loss(o, t)
        return dict(loss=loss)
