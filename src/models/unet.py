import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["unet_h"]


def unet_h(data_dim: tuple, n_blocks=4, base_channels=32, bilinear=True):
    n_channels = data_dim[0]
    return UNet(n_channels, n_channels, n_blocks, base_channels, bilinear)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of base_channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=4, base_channels=32, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)
        self.down_blocks = nn.ModuleList()
        for _ in range(n_blocks - 1):
            self.down_blocks.append(Down(base_channels, 2 * base_channels))
            base_channels *= 2
        factor = 1 if bilinear else 2
        self.down_blocks.append(Down(base_channels, factor * base_channels))

        self.up_blocks = nn.ModuleList()
        for _ in range(n_blocks - 1):
            base_channels //= 2
            self.up_blocks.append(Up(4 * base_channels, base_channels * factor, bilinear))
        self.up_blocks.append(Up(2 * base_channels, base_channels))
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)


    def forward(self, x):
        comp_stack = [self.inc(x)]
        for block in self.down_blocks:
            comp_stack.append(block(comp_stack[-1]))

        for block in self.up_blocks:
            x1 = comp_stack.pop()
            x2 = comp_stack.pop()
            comp_stack.append(block(x1, x2))

        logits = self.outc(comp_stack[0])
        return logits
