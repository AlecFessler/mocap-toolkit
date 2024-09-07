# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

from torch import nn
import torch.nn.init as init

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(
            self.conv.weight,
            mode='fan_in',
            nonlinearity='relu'
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
