from torch import nn
from torch.nn import init

from modules.ConvBlock import ConvBlock
from modules.SEBlock import SEBlock

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_block=None, se_block=None):
        super(ResBlock, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, drop_block=drop_block)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, drop_block=drop_block)
        if se_block is not None:
            self.se = SEBlock(se_block[0], se_block[1])

        self.init_weights()
        
    def init_weights(self):
        if hasattr(self, 'downsample'):
            init.kaiming_normal_(self.downsample[0].weight)

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out) if hasattr(self, 'se') else out
        out = out + residual
        return out