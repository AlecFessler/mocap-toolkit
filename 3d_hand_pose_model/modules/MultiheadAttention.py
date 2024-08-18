import math
import torch
from torch import nn
import torch.nn.init as init

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, dff, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        
        self.norm = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, dff)
        self.fc2 = nn.Linear(dff, embed_dim)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.query.weight)
        init.xavier_normal_(self.key.weight)
        init.xavier_normal_(self.value.weight)
        if isinstance(self.activation, nn.ReLU):
            init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        else:
            init.xavier_normal_(self.fc1.weight)
            init.xavier_normal_(self.fc2.weight)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        x = self.norm(x)

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = q.contiguous().view(-1, seq_len, self.head_dim)
        k = k.contiguous().view(-1, seq_len, self.head_dim)
        v = v.contiguous().view(-1, seq_len, self.head_dim)

        scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.nn.functional.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.bmm(weights, v)
        output = output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.norm2(output + x)

        output = self.fc1(output)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output + x