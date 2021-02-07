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
                d_model=2048):
        super().__init__()
        self.position_embed = PositionEmbeddingSine(hidden_dims//2)
        self.reshape = nn.Conv2d(d_model, hidden_dims, 1)
        self.transformer = Transformer(
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            num_encoder_layer=num_encoder_layer,
            num_decoder_layer=num_decoder_layer,
            dim_feed_forward=dim_feed_forward,
            dropout=dropout
        )
        self.adap = nn.AdaptiveAvgPool2d((None, 1))

        self.class_embed = nn.Linear(hidden_dims, 1)
        self.bbox_embed = MLP(input_dim=hidden_dims, hidden_dim=hidden_dims, output_dim=4, num_layers=3)

    def forward(self, template: Tensor, search: Tensor):
        pos_template, mask_template = self.position_embed(template)
        template = self.reshape(template)

        pos_search, mask_search = self.position_embed(search)
        search = self.reshape(search)

        out, out2 = self.transformer(template, mask_template, pos_template, search, mask_search, pos_search)
        # [6, 32, 16, 512]
        out = self.adap(out[-1].transpose(1,2)).flatten(1)
        out2 = self.adap(out2[-1].transpose(1,2)).flatten(1)

        outputs_class = self.class_embed(out).sigmoid()
        outputs_coord = self.bbox_embed(out2).sigmoid()
        
        return outputs_class, outputs_coord