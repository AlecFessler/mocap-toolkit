from torch import nn
import torch.nn.init as init

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.fc[0].weight, nonlinearity='relu')
        init.xavier_normal_(self.fc[2].weight)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # (b, c, 1, 1) -> (b, c)
        y = self.fc(y)  # (b, c) -> (b, c // reduction) -> (b, c)
        y = y.view(b, c, 1, 1)  # (b, c) -> (b, c, 1, 1)
        return x * y.expand_as(x)  # shape = (b, c, h, w) * (b, c, 1, 1) -> (b, c, h, w)