from torch import nn
import torch.nn.init as init

from modules.DropBlock import DropBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False, drop_block=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        if drop_block is not None:
            self.drop_block = DropBlock(**drop_block)

        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.drop_block(out) if hasattr(self, 'drop_block') else out
        return out