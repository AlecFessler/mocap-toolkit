# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch.nn as nn
import torch.nn.init as init

from SelfAttn import SelfAttn

class ConvSelfAttn(nn.Module):
    def __init__(
        self,
        channel_height,
        channel_width,
        embed_dim,
        num_heads,
        dropout=0.1
    ):
        super(ConvSelfAttn, self).__init__()
        self.encode = nn.Linear(channel_height * channel_width, embed_dim)
        self.attn = SelfAttn(embed_dim, num_heads, dropout)
        self.decode = nn.Linear(embed_dim, channel_height * channel_width)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.encode.weight)
        init.xavier_uniform_(self.decode.weight)

    def forward(self, x, mask=None):
        b, c, h, w = x.size()
        residual = x

        x = x.view(b, c, -1)
        x = self.encode(x)
        x = self.attn(x, mask)
        x = self.decode(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(b, c, h, w)
        return x + residual
