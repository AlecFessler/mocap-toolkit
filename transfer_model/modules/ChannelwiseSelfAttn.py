import torch.nn as nn
import torch.nn.init as init

from .MultiheadAttention import MultiheadAttention

class ChannelwiseSelfAttn(nn.Module):
    def __init__(self, channel_height, channel_width, embed_dim, dff, num_heads, dropout=0.1, weighted_channels_out=True, embeddings_sequence_out=True):
        super(ChannelwiseSelfAttn, self).__init__()
        self.dropout = nn.Dropout(dropout)

        assert weighted_channels_out or embeddings_sequence_out, "At least one of weighted_channels_out or embeddings_sequence_out should be True"

        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm((channel_height, channel_width))

        self.project_up = nn.Linear(channel_height * channel_width, embed_dim)
        self.attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dff=dff, dropout=dropout)
        
        self.project_down = nn.Linear(embed_dim, channel_height * channel_width) if weighted_channels_out else None
        
        self.embeddings_sequence = nn.Linear(embed_dim, embed_dim) if embeddings_sequence_out else None

    def init_weights(self):
        init.xavier_normal_(self.project_up.weight)
        if self.project_down:
            if isinstance(self.activation, nn.ReLU):
                init.kaiming_normal_(self.project_down.weight, nonlinearity='relu')
            else:
                init.xavier_normal_(self.project_down.weight)
        if self.embeddings_sequence:
            if isinstance(self.activation, nn.ReLU):
                init.kaiming_normal_(self.embeddings_sequence.weight, nonlinearity='relu')
            else:
                init.xavier_normal_(self.embeddings_sequence.weight)

    def forward(self, x):
        _, c, h, w = x.size()

        x = self.norm(x) # (B, C, H, W)

        residual = x # (B, C, H, W)

        x = x.view(x.size(0), x.size(1), -1) # (B, C, H * W)

        x = self.project_up(x) # (B, C, embed_dim)
        x = self.dropout(x)

        x = self.attn(x)

        if self.project_down:
            w_channels = self.project_down(x) # (B, C, H * W)
            w_channels = self.activation(w_channels)
            w_channels = self.dropout(w_channels)
            w_channels = w_channels.view(w_channels.size(0), c, h, w) # (B, C, H, W)
            w_channels = w_channels + residual
        
        if self.embeddings_sequence:
            emb_seq = self.embeddings_sequence(x) # (B, C, embed_dim)
            emb_seq = self.activation(emb_seq)
            emb_seq = self.dropout(emb_seq)

        if self.project_down and self.embeddings_sequence:
            return w_channels, emb_seq
        elif self.project_down:
            return w_channels
        else:
            return emb_seq