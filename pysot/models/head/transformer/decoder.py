from collections import namedtuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from typing import Optional

from pysot.models.head.transformer.utils import getClones, with_pos_embed

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super(TransformerDecoder, self).__init__()
        self.layers = getClones(decoder_layer, num_layers)

    def forward(self, 
                src2: Tensor, 
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                pos2: Optional[Tensor] = None):
        out = src2

        intermediate = []

        for layer in self.layers:
            out = layer(
                src2=src2,
                memory = memory,
                tgt_mask = tgt_mask,
                memory_mask = memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask = memory_key_padding_mask,
                pos = pos,
                pos2 = pos2
                )
            intermediate.append(out)

        return torch.stack(intermediate)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dims: int, num_heads:int, dropout: float, dim_feedforward: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dims, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dims)

        self.attn = nn.MultiheadAttention(hidden_dims, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dims)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dims, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dims),
            nn.Dropout(dropout),
        )

        self.norm_ff = nn.LayerNorm(hidden_dims)

    def forward(self, 
                src2: Tensor, 
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                pos2: Optional[Tensor] = None):
        q = k = with_pos_embed(src2, pos2)
        src2_, _ = self.self_attn(q, k, value=src2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        
        src2 = src2 + self.dropout1(src2_)
        src2 = self.norm1(src2)

        src2_, _ = self.attn(query=q,
                                   key=with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        src2 = src2 + self.dropout2(src2_)
        src2 = self.norm2(src2)
        src2_ = self.feed_forward(src2)
        src2 = src2 + src2_
        src2 = self.norm_ff(src2)
        return src2


