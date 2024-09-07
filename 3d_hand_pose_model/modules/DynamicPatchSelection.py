# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
import torch.nn.init as init

from ConvBlock import ConvBlock
from ConvSelfAttn import ConvSelfAttn

class DynamicPatchSelection(nn.Module):
    def __init__(
        self,
        in_channels,
        channel_height,
        channel_width,
        attn_embed_dim,
        attn_heads,
        pos_embed_dim,
        total_patches,
        patch_size,
        dropout=0.1
    ):
        super(DynamicPatchSelection, self).__init__()
        self.in_channels = in_channels
        self.total_patches = total_patches
        self.patch_size = patch_size

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=total_patches,
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2
        )
        self.attn = ConvSelfAttn(
            channel_height=channel_height,
            channel_width=channel_width,
            embed_dim=attn_embed_dim,
            num_heads=attn_heads,
            dropout=dropout
        )
        self.fc = nn.Linear(channel_height * channel_width, 2)
        self.activation = nn.Tanh()

        self.pos_embed = nn.Linear(2, pos_embed_dim)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.pos_embed.weight)

    def forward(self, x):
        b, c, h, w = x.size()
        features = self.conv(x)
        features = self.attn(features)
        features = features.view(b, self.total_patches, -1)

        # compute tensor of affine transform matrices
        # the outputs of the fc are the translation parameters
        # scaling factor is ratio of patch size to image size
        # no rotation is applied, left as 0s

        translation_params = self.fc(features).view(b, self.total_patches, 2)
        translation_params = self.activation(translation_params)

        affine_transforms = torch.zeros(b, self.total_patches, 2, 3, device=x.device)
        affine_transforms[:, :, 0, 0] = self.patch_size / w
        affine_transforms[:, :, 1, 1] = self.patch_size / h
        affine_transforms[:, :, :, 2] = translation_params

        grid = nn.functional.affine_grid(
            affine_transforms.view(-1, 2, 3),
            [b * self.total_patches, 1, self.patch_size, self.patch_size],
            align_corners=False
        )

        x = x.repeat(1, self.total_patches, 1, 1).view(b * self.total_patches, c, h, w)
        patches = nn.functional.grid_sample(
            x,
            grid,
            padding_mode='border',
            align_corners=False
        )
        patches = patches.view(
            b,
            self.total_patches,
            c * self.patch_size * self.patch_size,
        )

        pos_embeds = self.pos_embed(translation_params)

        return patches, pos_embeds
