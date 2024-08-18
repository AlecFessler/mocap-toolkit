from torch import nn
import torch

class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob):
        super(DropBlock, self).__init__()
        assert block_size % 2 == 1, "block_size must be odd"
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], 1, *x.shape[2:]) < gamma).float().to(x.device)

            block_mask = self._compute_block_mask(mask, x.shape)
            out = x * block_mask * block_mask.numel() / (block_mask.sum() + 1e-7)
            return out

    def _compute_gamma(self, x):
        p = 1 - self.keep_prob
        kernel_size = (self.block_size, self.block_size)
        feature_size = x.shape[2:]
        gamma = p * feature_size[0] * feature_size[1] / (kernel_size[0] * kernel_size[1]) / (((feature_size[0] - kernel_size[0] + 1) * (feature_size[1] - kernel_size[1] + 1)) + 1e-7)
        return gamma

    def _compute_block_mask(self, mask, x_shape):
        block_mask = 1 - nn.functional.max_pool2d(1 - mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        block_mask = nn.functional.interpolate(block_mask, size=x_shape[2:], mode='nearest') 
        return block_mask