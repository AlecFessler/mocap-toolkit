# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch.nn as nn
import torch.nn.init as init

from CrossAttn import CrossAttn

class ConvCrossAttn(nn.Module):
    def __init__(
        self,
        channel_height,
        channel_width,
        embed_dim,
        num_heads,
        dropout=0.1
    ):
        super(ConvCrossAttn, self).__init__()
        self.encode = nn.Linear(channel_height * channel_width, embed_dim)
        self.attn = CrossAttn(embed_dim, num_heads, dropout)
        self.decode = nn.Linear(embed_dim, channel_height * channel_width)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.encode.weight)
        init.xavier_uniform_(self.decode.weight)

    def forward(self, x, y, mask=None):
        b, c, h, w = x.size()
        residual = x

        x = x.view(b, c, -1)
        y = y.view(b, c, -1)
        x = self.encode(x)
        y = self.encode(y)
        x = self.attn(x, y, mask)
        x = self.decode(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(b, c, h, w)
        return x + residual
