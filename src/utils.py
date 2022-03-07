import functools
import inspect
import itertools

import numpy as np

import torch
from torch.distributions import MixtureSameFamily

from .distributions import TensorDatasetDistribution


def uniform_circle(n_samples: int) -> torch.Tensor:
    theta = 2 * torch.pi * torch.arange(n_samples) / n_samples
    x = torch.cos(theta)
    y = torch.sin(theta)

    return torch.stack([x, y]).T


def fibonacci_sphere(n_samples: int) -> torch.Tensor:
    phi = torch.pi * (3. - np.sqrt(5.))  # golden angle in radians
    y = 1 - 2 * torch.arange(n_samples) / (n_samples - 1)
    theta = phi * torch.arange(n_samples)  # golden angle increment
    radius = torch.sqrt(1 - y * y)
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    return torch.stack([x, y, z]).T


def get_explicit_P(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    P = torch.einsum("ij,ik->jk", y, x)
    return P / torch.norm(P)


def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        values = signature.bind(self, *args, **kwargs)
        for name, param in signature.parameters.items():
            if name == 'self':
                continue
            if name not in values.arguments:
                setattr(self, name, param.default)
            elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                setattr(self, name, values.arguments[name])
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                for name, val in values.arguments[name].items():
                    setattr(self, name, val)
        func(self, *args, **kwargs)
    return wrapper


def get_permutation(total_ndim: int, batch_ndim: int, sample_ndim: int) -> list:
    p = list(range(total_ndim))
    p[:batch_ndim], p[batch_ndim:batch_ndim+sample_ndim] = \
    p[sample_ndim:batch_ndim+sample_ndim], p[:sample_ndim]
    return p


def sample_from_tensordataset_classes(distribution: TensorDatasetDistribution,
                                      sample_shape) -> torch.Tensor:
    samples = []
    for target in distribution.classes:
        samples.append(distribution.sample_from_class(sample_shape, target))

    return torch.stack(samples)


def sample_from_gmm_components(gmm: MixtureSameFamily,
                               sample_shape) -> torch.Tensor:
    components = gmm._component_distribution
    samples = components.sample(sample_shape)

    permutation = get_permutation(samples.ndim,
                                  len(components.batch_shape),
                                  len(sample_shape))
    return samples.permute(permutation)


def sample_from_components(distribution, sample_shape) -> torch.Tensor:
    if isinstance(distribution, TensorDatasetDistribution):
        return sample_from_tensordataset_classes(distribution, sample_shape)
    # if isinstance(distribution, MixtureSameFamily):
    return sample_from_gmm_components(distribution, sample_shape)


def product_dict(dct):
    for values in itertools.product(*dct.values()):
        yield dict(zip(dct.keys(), values))
