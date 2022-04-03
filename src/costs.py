import geotorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as o


class CustomLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 weight_init=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._weight_size = self.weight.size()
        if weight_init is None:
            self.weight.data = self.weight.data.flatten()
        else:
            self.weight.data = weight_init.flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.view(self._weight_size), self.bias)


class DownBlock(nn.Module):
    def __init__(self, channels, factor=2, kernel_size=3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels,
                      factor * channels,
                      kernel_size,
                      stride=2,
                      padding=kernel_size//2,
                      bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(factor * channels)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)


class UpBlock(nn.Module):
    def __init__(self, channels, factor=2, kernel_size=3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(channels,
                               channels // factor,
                               kernel_size,
                               stride=2,
                               padding=kernel_size//2,
                               output_padding=kernel_size//2,
                               bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(channels // factor)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)


def _get_P(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    P = torch.einsum("ij,ik->jk", y.flatten(1), x.flatten(1))
    return P / torch.norm(P)


class InnerGW_opt:
    def __init__(self, p, q,
                 n_iter=10,
                 optimizer=o.Adam,
                 optimizer_params=dict(lr=5e-5),
                 init=None,
                 device=None) -> None:
        self.P = CustomLinear(p, q, bias=False, weight_init=init).to(device)
        geotorch.sphere(self.P, "weight")

        self.P_opt = optimizer(self.P.parameters(), **optimizer_params)
        self.n_iter = n_iter
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _x, _y = x.flatten(1), y.flatten(1)
        for _ in range(self.n_iter):
            self.P_opt.zero_grad()
            Px = self.P(_x)
            cost = torch.norm(Px - _y.detach(), dim=1) ** 2
            cost.mean().backward()
            self.P_opt.step()
        Px = self.P(_x)
        return torch.norm(Px - _y, dim=1) ** 2


class InnerGW_conv:
    def __init__(self,
                 depth=4, channels=16,
                 n_iter=10,
                 optimizer=o.Adam,
                 optimizer_params=dict(lr=5e-5),
                 device=None) -> None:
        layers = []
        for _ in range(1, depth):
            layers.append(DownBlock(channels))
            channels *= 2

        for _ in range(1, depth):
            layers.append(UpBlock(channels))
            channels //= 2

        self.P = nn.Sequential(
            nn.Conv2d(3, channels, 3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
            *layers,
            nn.ConvTranspose2d(channels, 3, 3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid()
        ).to(device)

        self.P_opt = optimizer(self.P.parameters(), **optimizer_params)
        self.n_iter = n_iter
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _y = y.flatten(1)
        for _ in range(self.n_iter):
            self.P_opt.zero_grad()
            Px = self.P(x).flatten(1)
            cost = torch.norm(Px - _y.detach(), dim=1) ** 2
            cost.mean().backward()
            self.P_opt.step()
        Px = self.P(x).flatten(1)

        return torch.norm(Px - _y, dim=1) ** 2


class InnerGW_const:
    def __init__(self, P) -> None:
        self.P = P

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _x, _y = x.flatten(1), y.flatten(1)
        Px = _x @ self.P.T
        return torch.norm(Px - _y, dim=1) ** 2


class SqGW_const:
    def __init__(self, P=None) -> None:
        self.P = P

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _x, _y = x.flatten(1), y.flatten(1)
        Px = _x @ self.P.T
        return torch.norm(Px - _y, dim=1) ** 2 - 2 * (_x.norm(dim=1) * _y.norm(dim=1)) ** 2


class SqGW_opt:
    def __init__(self, p, q,
                 n_iter=10,
                 optimizer=o.Adam,
                 optimizer_params=dict(lr=5e-5),
                 init=None,
                 device=None) -> None:
        self.P = CustomLinear(p, q, bias=False, weight_init=init).to(device)

        self.P_opt = optimizer(self.P.parameters(), **optimizer_params)
        self.n_iter = n_iter
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _x, _y = x.flatten(1), y.flatten(1)
        for _ in range(self.n_iter):
            self.P_opt.zero_grad()
            Px = self.P(_x)
            cost = torch.norm(Px - _y.detach(), dim=1) ** 2
            (cost.mean() + self.P.weight.norm() ** 2 / 4).backward()
            self.P_opt.step()
        Px = self.P(_x)
        return torch.norm(Px - _y, dim=1) ** 2 - 2 * (_x.norm(dim=1) * _y.norm(dim=1)) ** 2
