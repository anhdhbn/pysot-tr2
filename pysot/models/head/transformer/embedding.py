import math
from typing import Tuple

import torch
from torch import nn, Tensor


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_position_features: int = 64, temperature: int = 10000, normalize: bool = True,
                 scale: float = None):
        super(PositionEmbeddingSine, self).__init__()

        self.num_position_features = num_position_features
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        mask: provides specified elements in the key to be ignored by the attention.
              the positions with the value of True will be ignored
              while the position with the value of False will be unchanged.
              Since I am only training with images of the same shape, the mask should be all False.
              Modify the mask generation method if you would like to enable training with arbitrary shape.
        """
        N, _, H, W = x.shape

        mask = torch.zeros(N, H, W, dtype=torch.bool, device=x.device)
        # TODO generate mask
        not_mask = ~mask

        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)

        if self.normalize:
            epsilon = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + epsilon) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + epsilon) * self.scale

        dim_t = torch.arange(self.num_position_features, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_position_features)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), -1).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), -1).flatten(3)

        return torch.cat((pos_y, pos_x), 3).permute(0, 3, 1, 2), mask
