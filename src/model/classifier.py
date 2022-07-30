import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from sklearn.metrics import roc_auc_score
from timm.loss import LabelSmoothingCrossEntropy
from .convnext import convnext_tiny
from .layers import ListLayerNorm, ChangeKernel
from .transformer import PositionEmbedding1D, PositionEmbedding2D, Transformer


class Classifier(nn.Module):
    def __init__(
        self,
        num_class: int=5,
        num_queries: int=100,
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

        if (num_encoder_layers > 0) and (num_decoder_layers > 0):
            self.change = ChangeKernel(
                in_channels=num_kernels[-1],
                out_channels=d_model
            )

            self.pe_2d = PositionEmbedding2D(
                num_pos_feats=d_model//2,
                normalize=True
            )
            self.embed = nn.Embedding(
                num_embeddings=num_queries, 
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
                out_features=num_class
            )
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.constant_(self.classifier.bias, 0)
            
            self.with_transformer = True
        else:
            self.classifier = nn.Linear(
                in_features=num_kernels[-1],
                out_features=num_class
            )
            self.with_transformer = False

        self.ce_loss = LabelSmoothingCrossEntropy(label_smoothing)

    @staticmethod
    def bcwh2nbc(
        x: torch.Tensor
    ) -> torch.Tensor:
        return x.flatten(2).permute(2, 0, 1)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # convnext
        xs = self.convnext(x)
        xs = self.norm(xs)
        x = xs[-1]
        
        if self.with_transformer:
            # x, p
            x = self.change(x)
            p = self.pe_2d(x)

            x = self.bcwh2nbc(x)
            p = self.bcwh2nbc(p)

            if self.train:
                x = x + torch.randn_like(x)

            # y, q
            _, batch_size, _ = x.shape
            q = self.embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
            y = torch.zeros_like(q)

            # transformer
            o = self.transformer(x, p, y, q)
            o = torch.mean(o, axis=1)
        else:
            o = x.mean([-2, -1])

        o = self.classifier(o)
        return o

    def predict(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x = data["image"]
        return self(x)

    def loss(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x = data["image"]
        t = data["class_id"]

        # loss
        o = self(x)
        loss = self.ce_loss(o, t)

        # accuracy
        pred = o.detach().cpu().numpy()
        true = t.detach().cpu().numpy()
        pred_label = np.argmax(pred, axis=1)
        acc = torch.tensor(np.mean(true==pred_label))

        return dict(loss=loss, acc=acc)