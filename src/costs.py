import geotorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as o


class CustomLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, weight_init=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self._weight_size = self.weight.size()
        self.weight.data = self.weight.data.flatten() if weight_init is None else weight_init.flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.view(self._weight_size), self.bias)


def _get_explicit_P(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    P = torch.einsum("ij,ik->jk", y.flatten(1), x.flatten(1))
    return P / torch.norm(P)


class InnerGW_explicit:
    def __init__(self, l=.05):
        self.l = l

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        P = _get_explicit_P(x, y)
        return self.l * torch.norm(x.flatten(1) @ P.T - y.flatten(1), p="fro", dim=1) ** 2


class InnerGW_opt:
    def __init__(self, p, q, l=.05,
                 n_iter=10,
                 optimizer=o.Adam,
                 optimizer_params=dict(lr=5e-5),
                #  logger=None,
                 init=None,
                 device=None):
        self.P = nn.Sequential(
            nn.Flatten(),
            CustomLinear(p, q, bias=False, weight_init=init)
        ).to(device)
        geotorch.sphere(self.P[1], "weight")
        self.P_opt = optimizer(self.P.parameters(), **optimizer_params)

        self.l = l
        self.n_iter = n_iter

        # self.logger = logger
        self.step = 0
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_iter):
            self.P_opt.zero_grad()
            Px = self.P(x)
            cost = self.l * (torch.norm(Px - y.detach().flatten(1), p="fro", dim=1) ** 2)
            cost.mean().backward()
            self.P_opt.step()
        Px = self.P(x)
        # if self.logger:
        #     with torch.no_grad():
        #         target_P = _get_explicit_P(x, y).to(self.device)
        #         P_mse = F.mse_loss(self.P.weight, target_P)
        #         self.logger.log("P MSE", P_mse.item(), 1 + self.step // 15)
        #     self.step += 1
        return self.l * torch.norm(Px - y.flatten(1), p="fro", dim=1) ** 2


class InnerGW_conv:
    def __init__(self, l=.05,
                 n_iter=10,
                 optimizer=o.Adam,
                 optimizer_params=dict(lr=5e-5),
                #  logger=None,
                 device=None):
        self.P = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=7, padding=6, dilation=2, bias=False),
            nn.Conv2d(3, 3, kernel_size=7, padding=6, dilation=2, bias=False),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, kernel_size=7, padding=6, dilation=2, bias=False),
            nn.Conv2d(3, 3, kernel_size=7, padding=6, dilation=2, bias=False),
        ).to(device)
        geotorch.sphere(self.P[0], "weight")
        geotorch.sphere(self.P[1], "weight")
        geotorch.sphere(self.P[2], "weight")
        geotorch.sphere(self.P[3], "weight")
        self.P_opt = optimizer(self.P.parameters(), **optimizer_params)

        self.l = l
        self.n_iter = n_iter

        # self.logger = logger
        self.step = 0
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_iter):
            self.P_opt.zero_grad()
            Px = self.P(x).flatten(1)
            cost = self.l * (torch.norm(Px - y.detach().flatten(1), p="fro", dim=1) ** 2)
            cost.mean().backward()
            self.P_opt.step()
        Px = self.P(x).flatten(1)
        # if self.logger:
        #     with torch.no_grad():
        #         target_P = _get_explicit_P(x, y).to(self.device)
        #         P_mse = F.mse_loss(self.P.weight, target_P)
        #         self.logger.log("P MSE", P_mse.item(), 1 + self.step // 15)
        #     self.step += 1
        return self.l * torch.norm(Px - y.flatten(1), p="fro", dim=1) ** 2


class InnerGW_const:
    def __init__(self, P, l=.05):
        self.P = P
        self.l = l

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Px = x.flatten(1) @ self.P
        return self.l * torch.norm(Px - y.flatten(1), p="fro", dim=1) ** 2
