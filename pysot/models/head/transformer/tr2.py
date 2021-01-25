import torch
from torch.functional import Tensor
import torch.nn as nn

from pysot.models.head.transformer.embedding import PositionEmbeddingSine
from pysot.models.head.transformer.transformer import Transformer

class Tr2Head(nn.Module):
    def __init__(self,
                hidden_dims=512, 
                num_heads = 8, 
                num_encoder_layer=6, 
                num_decoder_layer=6, 
                dim_feed_forward=2048, 
                dropout=.1, 
                num_queries=10):
        super().__init__()
        self.position_embed = PositionEmbeddingSine(hidden_dims//2)
        self.reshape = nn.Conv2d(2048, hidden_dims, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dims)
        self.transformer = Transformer(
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            num_encoder_layer=num_encoder_layer,
            num_decoder_layer=num_decoder_layer,
            dim_feed_forward=dim_feed_forward,
            dropout=dropout
        )

    def forward(self, template: Tensor, search: Tensor):
        pos, mask = self.position_embed(template)
        features = self.reshape(template)

        pos2, mask2 = self.position_embed(search)
        features2 = self.reshape(search)

        out = self.transformer(features, mask, pos, features2, mask2, pos2)
        print(features.shape, pos.shape, mask.shape)
        print(features2.shape, pos2.shape, mask2.shape)
        print(out.shape)