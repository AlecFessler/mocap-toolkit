import torch.nn as nn
import torch.nn.init as init

class OctaveConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0):
        super(OctaveConvBlock, self).__init__()
        self.alpha = alpha
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        low_channels_in = int(in_channels * alpha)
        high_channels_in = in_channels - low_channels_in
        low_channels_out = int(out_channels * alpha)
        high_channels_out = out_channels - low_channels_out

        self.conv_l2h = nn.Conv2d(low_channels_in, out_channels, kernel_size, stride, padding)
        self.conv_l2l = nn.Conv2d(low_channels_in, low_channels_out, kernel_size, stride, padding)
        self.conv_h2l = nn.Conv2d(high_channels_in, low_channels_out, kernel_size, stride, padding)
        self.conv_h2h = nn.Conv2d(high_channels_in, high_channels_out, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.conv_l2h.weight)
        init.kaiming_normal_(self.conv_l2l.weight)
        init.kaiming_normal_(self.conv_h2l.weight)
        init.kaiming_normal_(self.conv_h2h.weight)

    def forward(self, x):
        high_channels_in = x.size(1) - int(x.size(1) * self.alpha)
        x_h = x[:, :high_channels_in, :, :]
        x_l = x[:, high_channels_in:, :, :]
        
        x_h = self.conv_h2h(x_h)
        
        x_l = self.downsample(x_l)
        x_l = self.conv_h2l(x_l)
        x_l = self.upsample(x_l)
        
        out = x_h + x_l
        
        out = self.bn(out)
        out = self.activation(out)

        return out
