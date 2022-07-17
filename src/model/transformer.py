import copy
import math
import torch
import torch.nn as nn
from typing import Optional


class PositionEmbedding1D(nn.Module):
    def __init__(
        self,
        temperature: int=10000
    ) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        seq_len, batch_size, d_model = x.shape
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device) * (-math.log(self.temperature)/d_model)
        )
        pe = torch.zeros(seq_len, 1, d_model, device=x.device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe
    

class PositionEmbedding2D(nn.Module):
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
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        d_ff: int, 
        dropout: float=0.1
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor
    ) -> torch.Tensor:
        query = key = x + p
        value = x
        a, _ = self.attention(query, key, value)
        b = self.norm_1(x+self.dropout_1(a))
        o = self.norm_2(b+self.dropout_2(self.ffn(b)))
        return o
    
    
class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        encoder_layer: TransformerEncoderLayer, 
        num_layers: int
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(
        self, 
        x: torch.Tensor,
        p: torch.Tensor
    ) -> torch.Tensor:
        output = x
        for layer in self.layers:
            output = layer(output, p)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        d_ff: int, 
        dropout: float=0.1
    ) -> None:
        super().__init__()
        self.attention_1 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attention_2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        query = key = y + q
        value = y        
        a, _ = self.attention_1(query, key, value)
        b = self.norm_1(y+self.dropout_1(a))
        
        query = b + q
        key = x + p
        value = x
        c, _ = self.attention_2(query, key, value)
        d = self.norm_2(b+self.dropout_2(c))        
        o = self.norm_3(d+self.dropout_3(self.ffn(d)))
        return o
    
    
class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer: TransformerDecoderLayer, 
        num_layers: int
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(
        self, 
        x: torch.Tensor,
        p: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        output = y
        for layer in self.layers:
            output = layer(x, p, output, q)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
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


class Transformer(nn.Module):
    def __init__(
        self, 
        d_model: int=512,
        n_head: int=8, 
        num_encoder_layers: int=6,
        num_decoder_layers: int=6, 
        d_ff: int=2048, 
        dropout: float=0.1
    ) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            n_head=n_head, 
            d_ff=d_ff, 
            dropout=dropout
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_encoder_layers
        )

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, 
            n_head=n_head,
            d_ff=d_ff,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer, 
            num_layers=num_decoder_layers
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, 
        x: torch.Tensor,
        p: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """
        x: Height x Width, Batch, N_hidden
        p: Height x Width, Batch, N_hidden
        y: Sequence, Batch, N_hidden
        q: Sequence, Batch, N_hidden
        out: Batch, Sequcne, N_hidden
        """
        memory = self.encoder(x, p)
        out = self.decoder(memory, p, y, q)
        return out.permute(1, 0, 2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
