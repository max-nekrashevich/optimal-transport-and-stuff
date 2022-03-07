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
           "ImageDistribution",
           "MoonsDistribution",
           "DatasetDistribution",
           "CurveDistribution",
           "TensorDatasetDistribution",
           "gaussian_mixture",
           "sample_from_gmm_components"]


def get_permutation(total_ndim: int, batch_ndim: int, sample_ndim: int) -> list:
    p = list(range(total_ndim))
    p[:batch_ndim], p[batch_ndim:batch_ndim+sample_ndim] = \
    p[sample_ndim:batch_ndim+sample_ndim], p[:sample_ndim]
    return p


def gaussian_mixture(locs, scales=None, probs=None) -> d.MixtureSameFamily:
    if scales is None:
        scales = torch.ones_like(locs)
    if probs is None:
        probs = torch.ones(locs.size(0)).type_as(locs)
    mix = d.Categorical(probs)
    comp = d.Independent(d.Normal(locs, scales), 1)
    return d.MixtureSameFamily(mix, comp)


def sample_from_gmm_components(gmm: d.MixtureSameFamily, sample_shape):
    components = gmm._component_distribution
    samples = components.sample(sample_shape)

    permutation = get_permutation(samples.ndim,
                                  len(components.batch_shape),
                                  len(sample_shape))
    return samples.permute(permutation)


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

    return gaussian_mixture(locs, scales, probs)


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
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def sample(self, sample_shape):
        ix = torch.randint(0, self.features.size(0), sample_shape)
        return self.features[ix]

    def sample_from_class(self, sample_shape, target):
        class_indices = torch.where(self.targets == target)[0]
        ix = class_indices[torch.randint(0, class_indices.size(0), sample_shape)]
        return self.features[ix]


class CurveDistribution:
    def __init__(self, curve):
        self.curve = curve
        self.t_distribution = d.Uniform(0, 1)

    def sample(self, sample_shape):
        t_samples = self.t_distribution.sample(sample_shape)
        return self.curve(t_samples)
