import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.head.transformer.embedding import PositionEmbeddingSine
from pysot.models.head.transformer.transformer import Transformer

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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
        self.transformer = Transformer(
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            num_encoder_layer=num_encoder_layer,
            num_decoder_layer=num_decoder_layer,
            dim_feed_forward=dim_feed_forward,
            dropout=dropout
        )
        self.class_embed = nn.Linear(hidden_dims, 1)
        self.bbox_embed = MLP(input_dim=hidden_dims, hidden_dim=hidden_dims, output_dim=4, num_layers=3)

    def forward(self, template: Tensor, search: Tensor):
        pos, mask = self.position_embed(template)
        features = self.reshape(template)

        pos2, mask2 = self.position_embed(search)
        features2 = self.reshape(search)

        out = self.transformer(features, mask, pos, features2, mask2, pos2)
        # print(out.shape)
        # outputs_coord = self.bbox_embed(out).sigmoid()
        # print(outputs_coord.shape)
        # N = out.size(0)
        # out = out.view((N, -1))

        # self.class_embed = nn.Linear(out.size(1), 1)
        # self.bbox_embed = MLP(input_dim=out.size(1), hidden_dim=256, output_dim=4, num_layers=3)
        # self.bbox_embed = nn.Linear(out.size(1), 4)
        outputs_class = self.class_embed(out).sigmoid()
        outputs_coord = self.bbox_embed(out).sigmoid()

        print(outputs_class.shape, outputs_coord.shape)
        # return outputs_class, outputs_coord