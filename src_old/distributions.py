import torch
import torch.distributions as d

from numpy import prod
from sklearn.datasets import make_moons
from torch.distributions import MultivariateNormal

from torchvision.datasets import MNIST
from torch.utils.data import RandomSampler, DataLoader
import torchvision.transforms as t


__all__ = ["Uniform",
           "Normal",
           "MultivariateNormal",
           "image_uniform",
           "MoonDistribution",
           "DatasetDistribution",
           "CurveDistribution",
           "TensorDatasetDistribution",
           "GaussianMixture"]


class Uniform(d.Uniform):
    def log_prob(self, value):
        mask = torch.flatten((self.low < value) & (value < self.high), -self.low.ndim).all(-1)
        log_probs = torch.empty(mask.size()).fill_(-torch.inf).type_as(self.low)
        log_probs[mask] = super().log_prob(value[mask]).flatten(-self.low.ndim).sum(-1)
        return log_probs


class Normal(d.Normal):
    def log_prob(self, value):
        return super().log_prob(value).flatten(-self.loc.ndim).sum(-1)



def _clip(mask):
    i, j = torch.where(mask)
    return mask[i.min():i.max(), j.min():j.max()]



class GaussianMixture(d.MixtureSameFamily):
    def __init__(self, locs, scales=None, probs=None):
        if scales is None:
            scales = torch.ones_like(locs)
        if probs is None:
            probs = torch.ones(locs.size(0)).type_as(locs)
        mixture_distribution = d.Categorical(probs)
        component_distribution = d.Independent(d.Normal(locs, scales), 1)
        super().__init__(mixture_distribution, component_distribution)


def image_uniform(image_tensor, scale, center=None, sigma=.01, n_components=1000):
    image_tensor = _clip(image_tensor).flip(0)
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


class MoonDistribution:
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


class DatasetDistribution:
    def __init__(self, dataset, num_workers=2):
        self.dataset = dataset
        self.n_samples = None
        self.num_workers = num_workers

    def _update_sampler(self):
        self.sampler = RandomSampler(self.dataset,
                                     replacement=True,
                                     num_samples=self.n_samples)

        self.loader = DataLoader(self.dataset,
                                 batch_size=self.n_samples,
                                 sampler=self.sampler,
                                 num_workers=self.num_workers)

    def sample(self, sample_shape, return_labels=False):
        n_samples = int(prod(sample_shape))
        if self.n_samples != n_samples:
            self.n_samples = n_samples
            self._update_sampler()

        samples, labels = next(iter(self.loader))
        samples = samples.view(*sample_shape, *samples.shape[1:])

        if return_labels:
            return samples, labels
        return samples


class TensorDatasetDistribution:
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        self.classes = labels.unique()

    def sample(self, sample_shape, *, component=None, return_labels=False):
        if component is not None:
            component_indices = torch.where(self.labels == component)[0]
            ix = component_indices[torch.randint(0, component_indices.size(0), sample_shape)]
        else:
            ix = torch.randint(0, self.features.size(0), sample_shape)

        if return_labels:
            return self.features[ix], self.labels[ix]
        return self.features[ix]


class CurveDistribution:
    def __init__(self, curve):
        self.curve = curve
        self.t_distribution = d.Uniform(0, 1)

    def sample(self, sample_shape):
        t_samples = self.t_distribution.sample(sample_shape)
        return self.curve(t_samples)
