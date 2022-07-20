import torch

from .base import BasicDistribution, Uniform, _to_tensor


# TODO: Implement splines


__all__ = ["Curve",
           "CircleCurve",
           "LineCurve",
           "CurveDistribution"]


class CurveDistribution(BasicDistribution):
    def __init__(self, curve, *, device=None):
        self.curve = curve
        self.t_distribution = Uniform(0., 1., device=device)
        super().__init__(curve(torch.tensor(0.)).size(), device)

    def sample(self, sample_shape=torch.Size(), **kwargs):
        t_samples = self.t_distribution.sample(sample_shape)
        return self.curve(t_samples)

    @property
    def mean(self):
        t = torch.linspace(0., 1., device=self.device)
        return self.curve(t).mean(0)


class Curve:
    def __init__(self, device) -> None:
        self.device = device

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device):
        self.device = device


class CircleCurve(Curve):
    def __init__(self, center=0., radius=1., *, device):
        self.center = _to_tensor(center, device)
        self.radius = _to_tensor(radius, device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        x = torch.cos(2 * torch.pi * t) * self.radius + self.center
        y = torch.sin(2 * torch.pi * t) * self.radius + self.center
        return torch.stack([x, y], dim=-1)

    def to(self, device):
        super().to(device)
        self.center.to(device)
        self.radius.to(device)


class LineCurve(Curve):
    def __init__(self, start=(-1., 0.), end=(1., 0.), *, device):
        self.start = _to_tensor(start, device)
        self.end = _to_tensor(end, device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        x = t * (self.end[0] - self.start[0]) + self.start[0]
        y = t * (self.end[1] - self.start[1]) + self.start[1]
        return torch.stack([x, y], dim=-1)

    def to(self, device):
        super().to(device)
        self.start.to(device)
        self.end.to(device)