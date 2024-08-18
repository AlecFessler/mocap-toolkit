import torch
from torch import nn
from torch.nn import init

class PosEncodings(nn.Module):
    def __init__(self, dims, mean=0.0, std=0.02, dropout=0.1):
        super(PosEncodings, self).__init__()
        self.mean = mean
        self.std = std
        self.encodings = nn.Parameter(torch.zeros(dims), requires_grad=True)
        self.norm = nn.LayerNorm(dims)
        self.dropout = nn.Dropout(p=dropout)
        
        self.init_weights()

    def init_weights(self):
        init.normal_(self.encodings, mean=self.mean, std=self.std)

    def forward(self, x):
        out = x + self.encodings
        out = self.norm(out)
        out = self.dropout(out)
        return out