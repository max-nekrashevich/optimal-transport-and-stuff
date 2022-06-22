import typing as tp

import geotorch
import torch
import torch.nn as nn
import torch.nn.functional as F


def kernel_1(x: torch.Tensor, y: torch.Tensor, outer=False) -> torch.Tensor:
    x, y = x.flatten(1), y.flatten(1)
    if outer or x.size(0) != y.size(0):
        x, y = x.unsqueeze(1), y.unsqueeze(0)
    return x.norm(dim=-1) + y.norm(dim=-1) - .5 * (x - y).norm(dim=-1)


def kernel_2(x: torch.Tensor, y: torch.Tensor, outer=False) -> torch.Tensor:
    x, y = x.flatten(1), y.flatten(1)
    if outer or x.size(0) != y.size(0):
        x, y = x.unsqueeze(1), y.unsqueeze(0)
    return torch.einsum("...i,...i->...", x, y)


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


class DownBlock(nn.Sequential):
    def __init__(self, channels, factor=2, kernel_size=3) -> None:
        super().__init__(
            nn.Conv2d(channels,
                      factor * channels,
                      kernel_size,
                      stride=2,
                      padding=kernel_size//2,
                      bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(factor * channels)
        )


class UpBlock(nn.Sequential):
    def __init__(self, channels, factor=2, kernel_size=3) -> None:
        super().__init__(
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


def _get_P(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    P = torch.einsum("ij,ik->jk", y.flatten(1), x.flatten(1))
    return P / torch.norm(P)


class Cost(nn.Module):
    def get_functional(self,
                       x: torch.Tensor,
                       x_prime: torch.Tensor,
                       mover: nn.Module) -> torch.Tensor:
        raise NotImplementedError()


class InnerGW_base(Cost):
    def __init__(self, kernel) -> None:
        super().__init__()
        self.kernel = kernel

    @torch.no_grad()
    def get_functional(self, x: torch.Tensor, x_prime: torch.Tensor, mover: nn.Module) -> torch.Tensor:
        h_x = mover(x)
        h_x_prime = mover(x_prime)
        return torch.mean((self.kernel(x, x_prime, outer=True) -
                           self.kernel(h_x, h_x_prime, outer=True)) ** 2)


class InnerGW(InnerGW_base):
    def __init__(self, p, q, gamma=None, init=None, device=None) -> None:
        super().__init__(kernel_2)
        if init is not None:
            self.P = init
        elif p != q:
            self.P = torch.eye(q, p, device=device)
        else:
            self.P = None
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = x.flatten(1), y.flatten(1)
        if self.gamma is not None:
            self.P *= (1 - self.gamma)
            P_update = (y.detach().T @ x.detach())
            self.P += self.gamma * P_update / torch.norm(P_update)
        if self.P is not None:
            Px = x @ self.P.T
        else:
            Px = x
        return F.mse_loss(Px, y)


class InnerGW_opt(InnerGW_base):
    def __init__(self, p, q, init=None, device=None) -> None:
        super().__init__(kernel_2)
        self.P = CustomLinear(p, q, bias=False, weight_init=init).to(device)
        geotorch.sphere(self.P, "weight")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = x.flatten(1), y.flatten(1)
        Px = self.P(x)
        return F.mse_loss(Px, y)


class InnerGW_bottleneck(InnerGW_base):
    def __init__(self, depth=4, channels=16, device=None) -> None:
        super().__init__(kernel_2)
        layers: tp.List[nn.Module] = []
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Px = self.P(x)
        return F.mse_loss(Px, y)


class InnerGW_conv(InnerGW_base):
    def __init__(self, channels=16, depth=4, device=None) -> None:
        super().__init__(kernel_2)
        layers = [
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
            for _ in range(2, depth)
        ]
        self.P = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, channels, 3, padding=1, bias=False)),
            *layers,
            nn.utils.spectral_norm(nn.Conv2d(channels, 3, 3, padding=1, bias=False)),
        ).to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Px = self.P(x)
        return F.mse_loss(Px, y)


# class SqGW(nn.Module):
#     def __init__(self, P) -> None:
#         super().__init__()
#         self.P = P

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         x, y = x.flatten(1), y.flatten(1)
#         Px = x @ self.P.T
#         return F.mse_loss(Px, y) - 2 * (x.norm(dim=1) * y.norm(dim=1)) ** 2


# class SqGW_opt(nn.Module):
#     def __init__(self, p, q,
#                  init=None,
#                  device=None) -> None:
#         super().__init__()
#         self.P = CustomLinear(p, q, bias=False, weight_init=init).to(device)

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         x, y = x.flatten(1), y.flatten(1)
#         Px = self.P(x)
#         return F.mse_loss(Px, y) - 2 * (x.norm(dim=1) * y.norm(dim=1)) ** 2


class innerGW_kernel(InnerGW_base):
    def __init__(self, kernel, source, mover, n_samples_mc=5) -> None:
        super().__init__(kernel)
        self._mover = [mover]  # Needed to prevent registering as a submodule
        self.kernel = kernel
        self.source = source
        self.n = n_samples_mc

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x1 = self.source.sample((self.n,))
        y1 = self._mover[0](x1)

        x2 = self.source.sample((self.n,))
        y2 = self._mover[0](x2)

        k_y_y = self.kernel(y, y)
        k_x_x1 = self.kernel(x, x1, outer=True)
        k_y_y1 = self.kernel(x, x1, outer=True)
        k_x2_x = self.kernel(x2, x, outer=True)
        k_y1_y2 = self.kernel(y1, y2, outer=True)

        cost = k_y_y - 2 * torch.mean(k_x_x1 * k_y_y1, dim=-1) + \
            torch.einsum("bi,ij,jb->b", k_x_x1, k_y1_y2, k_x2_x) / self.n ** 2

        return cost
