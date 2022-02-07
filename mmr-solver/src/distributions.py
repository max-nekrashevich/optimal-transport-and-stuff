import torch
import torch.distributions as d

from numpy import prod
from sklearn.datasets import make_moons
from torch.distributions import MultivariateNormal


__all__ = ["Uniform",
           "Normal",
           "MultivariateNormal",
           "GaussianMixture",
           "ImageDistribution",
           "MoonsDistribution"]


def clip(tensor):
    i, j = torch.where(tensor)
    return tensor[i.min():i.max(), j.min():j.max()]


class Uniform(d.Uniform):
    def log_prob(self, value):
        mask = torch.flatten((self.low < value) & (value < self.high), -self.low.ndim).all(-1)
        log_probs = torch.empty(mask.size()).fill_(-torch.inf).type_as(self.low)
        log_probs[mask] = super().log_prob(value[mask]).flatten(-self.low.ndim).sum(-1)
        return log_probs


class Normal(d.Normal):
    def log_prob(self, value):
        return super().log_prob(value).flatten(-self.loc.ndim).sum(-1)


def GaussianMixture(locs, scales=None, probs=None) -> d.Distribution:
    if scales is None:
        scales = torch.ones_like(locs)
    if probs is None:
        probs = torch.ones(locs.size(0)).type_as(locs)
    mix = d.Categorical(probs)
    comp = d.Independent(d.Normal(locs, scales), 1)
    return d.MixtureSameFamily(mix, comp)


def ImageDistribution(image_tensor, scale, center=None, sigma=.01, n_components=1000):
    image_tensor = clip(image_tensor).flip(0)
    density = image_tensor / image_tensor.sum()
    nonzero = torch.stack(torch.where(density), dim=-1)
    scale = torch.tensor(scale)
    size = torch.tensor(image_tensor.size())

    if center is None:
        center = torch.zeros(2)
    center = center

    n_components = min(n_components, torch.count_nonzero(density))

    ix = torch.multinomial(density[density != 0], n_components)

    locs = center + scale * (2 * nonzero[ix] / size - 1).flip(1)
    scales = torch.empty_like(locs).fill_(sigma)
    probs = density[density != 0][ix]

    return GaussianMixture(locs, scales, probs)


class MoonsDistribution:
    def __init__(self, upper=False, scale=1., center=None, sigma=.01):
        self.upper = upper
        self.scale = torch.tensor(scale)
        if center is None:
            center = torch.zeros(2)
        self.center = center
        self.sigma = sigma

    def sample(self, sample_shape):
        n_samples = prod(sample_shape, dtype=int)
        points, _ = make_moons((n_samples, 0) if self.upper else (0, n_samples),
                               noise=self.sigma)
        points = torch.from_numpy(points).float()
        points = self.center + self.scale * points
        return points.view(*sample_shape, -1)
