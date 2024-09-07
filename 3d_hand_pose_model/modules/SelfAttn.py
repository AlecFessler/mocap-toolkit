# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

from torch import nn
import torch.nn.init as init

class SelfAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1
    ):
        super(SelfAttn, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, f"embed_dim must be divisible by num_heads"

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, mask=None):
        x = self.norm(x)
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        attn_output = self.dropout(attn_output)

        x = x + attn_output
        x = self.norm(x)

        x_ff = self.fc1(x)
        x_ff = self.activation(x_ff)
        x_ff = self.fc2(x_ff)
        x_ff = self.dropout(x_ff)

        return x + x_ff
