import geotorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as o


def _get_explicit_P(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    P = torch.einsum("ij,ik->jk", y, x)
    return P / torch.norm(P)


class InnerGW_opt:
    def __init__(self, p, q, l=.05,
                 n_iter=10,
                 optimizer=o.Adam,
                 optimizer_params=dict(lr=5e-5),
                #  logger=None,
                 device=None):
        self.P = nn.Sequential(
            nn.Flatten(),
            nn.Linear(p, q, bias=False)
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
            cost = self.l * (torch.norm(Px - y.detach(), p="fro", dim=1) ** 2)
            cost.mean().backward()
            self.P_opt.step()
        Px = self.P(x)
        # if self.logger:
        #     with torch.no_grad():
        #         target_P = _get_explicit_P(x, y).to(self.device)
        #         P_mse = F.mse_loss(self.P.weight, target_P)
        #         self.logger.log("P MSE", P_mse.item(), 1 + self.step // 15)
        #     self.step += 1
        return self.l * torch.norm(Px - y, p="fro", dim=1) ** 2


class InnerGW_const:
    def __init__(self, P, l=.05):
        self.P = P
        self.l = l

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Px = x.flatten(1) @ self.P
        return self.l * torch.norm(Px - y, p="fro", dim=1) ** 2
