import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["unet_h"]


def unet_h(data_dim: tuple, n_blocks=4, base_channels=32, bilinear=True):
    n_channels = data_dim[0]
    return UNet(n_channels, n_channels, n_blocks, base_channels, bilinear)


class DoubleConv(nn.Sequential):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if not mid_channels:
            mid_channels = out_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )


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
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

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

        self.in_conv = DoubleConv(in_channels, base_channels)

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

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        comp_stack = [self.in_conv(x)]
        for block in self.down_blocks:
            comp_stack.append(block(comp_stack[-1]))

        x = comp_stack.pop()
        for block in self.up_blocks:
            x = block(x, comp_stack.pop())

        return self.out_conv(x)
