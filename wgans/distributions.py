import torch


class Distribution:
    def sample(self, n: int = 1) -> torch.Tensor:
        raise NotImplementedError


class Normal:
    def __init__(self, dim, loc=None, scale=None):
        loc = loc or torch.zeros(dim)
        scale = scale or torch.ones(dim)
        self.distribution = torch.distributions.Normal(loc, scale)

    def sample(self, n: int = 1) -> torch.Tensor:
        return self.distribution.sample((n,))
