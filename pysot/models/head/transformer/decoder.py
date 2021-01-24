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
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        out = tgt

        intermediate = []

        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
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
                tgt: Tensor, 
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = k = with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2, _ = self.attn(query=with_pos_embed(tgt, query_pos),
                                   key=with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + tgt2
        tgt = self.norm_ff(tgt)
        return tgt


