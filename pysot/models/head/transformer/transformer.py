import torch
import torch.nn as nn
from torch import Tensor

from pysot.models.head.transformer.encoder import TransformerEncoder, TransformerEncoderLayer
from pysot.models.head.transformer.decoder import TransformerDecoder, TransformerDecoderLayer

class Transformer(nn.Module):
    def __init__(self,
                hidden_dims=512, 
                num_heads = 8, 
                num_encoder_layer=6, 
                num_decoder_layer=6, 
                dim_feed_forward=2048, 
                dropout=.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feed_forward
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layer
        )

        decoder_layer = TransformerDecoderLayer(
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feed_forward
        )

        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layer)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, query: Tensor, pos: Tensor, src2: Tensor) -> Tensor:
        """
        :param src: tensor of shape [batchSize, hiddenDims, imageHeight // 32, imageWidth // 32]

        :param mask: tensor of shape [batchSize, imageHeight // 32, imageWidth // 32]
                     Please refer to detr.py for more detailed description.

        :param query: object queries, tensor of shape [numQuery, hiddenDims].

        :param pos: positional encoding, the same shape as src.

        :return: tensor of shape [batchSize, numQuery * numDecoderLayer, hiddenDims]
        """
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) # HWxNxC

        mask = mask.flatten(1) # NxHW
        pos = pos.flatten(2).permute(2, 0, 1) # HWxNxC
        query = query.unsqueeze(1).repeat(1, bs, 1) # num_queries * N * h_dims
        tgt = torch.zeros_like(src2)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos)
        out = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos, query_pos=src2) # num_decoder_layer x num_queries x N x C 
        return out.transpose(0, 2).flatten(1, 2)#, memory.permute(1, 2, 0).view(bs, c, h, w)