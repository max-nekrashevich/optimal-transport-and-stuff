import copy
import functools
import inspect

import numpy as np
import torch
import torchvision.datasets as datasets

from tqdm.auto import tqdm


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


def load_mnist(root, transform=None, train=True, verbose=True):
    dataset = datasets.MNIST(root, transform=transform, download=True, train=train)
    images, targets = [], []
    for image, target in tqdm(dataset, disable=not verbose):
        images.append(image)
        targets.append(target)
    return torch.stack(images), torch.tensor(targets)


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


def copy_models(*models):
    return [copy.deepcopy(model) for model in models]
