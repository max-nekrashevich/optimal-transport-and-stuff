import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["resnet14_d", "resnet18_d", "resnet34_d",
           "resnet14_g", "resnet18_g", "resnet34_g"]


def resnet14_d(data_dim: tuple, code_size: int = 1):
    return ResNetEncoder([1, 1, 1, 1], data_dim, code_size)

def resnet18_d(data_dim: tuple, code_size: int = 1):
    return ResNetEncoder([2, 2, 2, 2], data_dim, code_size)

def resnet34_d(data_dim: tuple, code_size: int = 1):
    return ResNetEncoder([3, 4, 6, 3], data_dim, code_size)

def resnet14_g(data_dim: tuple, code_size: int):
    return ResNetDecoder([1, 1, 1, 1], data_dim, code_size)

def resnet18_g(data_dim: tuple, code_size: int):
    return ResNetDecoder([2, 2, 2, 2], data_dim, code_size)

def resnet34_g(data_dim: tuple, code_size: int):
    return ResNetDecoder([3, 4, 6, 3], data_dim, code_size)


class EncoderBlock(nn.Module):

    def __init__(self, inplanes: int, planes: int,
                 stride: int = 1, downsample = None) -> None:

        super().__init__()

        self.downsample = downsample
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layers(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, layers, data_dim=(3, 64, 64), code_size=1000):
        super().__init__()
        self.inplanes = 64
        self.data_dim = data_dim
        self.code_size = code_size
        C_in, H_in, W_in = data_dim
        self.layers = nn.Sequential(
            nn.Conv2d(C_in, 64, kernel_size=7,
                                stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, layers[0]),
            self._make_layer(128, layers[1], stride=2),
            self._make_layer(256, layers[2], stride=2),
            self._make_layer(512, layers[3], stride=2),
            nn.AvgPool2d(kernel_size=(H_in // 32, W_in // 32)),
            nn.Flatten(),
            nn.Linear(512, code_size)
        )


    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(EncoderBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(EncoderBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):

    def __init__(self, inplanes: int, planes: int,
                 stride: int = 1, upsample = None) -> None:

        super().__init__()

        self.upsample = upsample
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=stride, mode="bilinear",
                        align_corners=True),
            nn.Conv2d(inplanes, planes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layers(x)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = F.relu(out)
        return out


class ResNetDecoder(nn.Module):
    def __init__(self, layers, data_dim=(3, 64, 64), code_size=1000):
        super().__init__()
        self.inplanes = 512
        self.data_dim = data_dim
        self.code_size = code_size
        C_out, H_out, W_out = data_dim
        self.layers = nn.Sequential(
            nn.Linear(code_size, 512),
            nn.Unflatten(1, (512, 1, 1)),
            nn.Upsample(scale_factor=(H_out // 32, W_out // 32),
                        mode="bilinear", align_corners=True),
            self._make_layer(256, layers[0], stride=2),
            self._make_layer(128, layers[1], stride=2),
            self._make_layer(64, layers[2], stride=2),
            self._make_layer(64, layers[3]),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ConvTranspose2d(64, C_out, kernel_size=8,
                                         stride=2, padding=3, bias=False),
            nn.BatchNorm2d(C_out),
            nn.Sigmoid(),
        )


    def _make_layer(self, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=True),
                nn.Conv2d(self.inplanes, planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        for _ in range(blocks - 1):
            layers.append(DecoderBlock(self.inplanes, self.inplanes))
        layers.append(DecoderBlock(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
