import copy
import functools
import inspect
import itertools
import pathlib

import h5py
import numpy as np
import torch

from scipy import linalg
from torchvision import transforms as t
from tqdm.auto import tqdm

from .models.inception import InceptionV3


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


def load_dataset(dataset_class, verbose=True, **dataset_params):
    dataset = dataset_class(download=True, **dataset_params)
    images, targets = [], []
    for image, target in tqdm(dataset, disable=not verbose):
        images.append(image)
        targets.append(target)
    return torch.stack(images), torch.tensor(targets)


def load_h5py(path, key=None, transform=t.ToTensor(), verbose=True):
    file = h5py.File(path, "r")
    if key is None:
        key = next(iter(file.keys()))
    images = []
    for image in tqdm(file.get(key), disable=not verbose):
        images.append(transform(image))
    return torch.stack(images)


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


def load_models(path, verbose=True, map_location=None, **models):
    path = pathlib.Path(path)
    if verbose:
        print("Loading models from", path)
    for model_name, model in models.items():
        model.load_state_dict(torch.load(f"{path}/{model_name}.torch",
                                         map_location=map_location))


def save_models(path, verbose=True, **models):
    path = pathlib.Path(path)
    if not path.exists():
        path.mkdir()
    if verbose:
        print("Saving models to", path)
    for model_name, model in models.items():
        torch.save(model.state_dict(), f"{path}/{model_name}.torch")


def nwise(iterable, n=2):
    iters = itertools.tee(iterable, n)
    for i, it in enumerate(iters):
        next(itertools.islice(it, i, i), None)
    return zip(*iters)


def _init_opt_or_sch(type, params, init_arg, base_type, base_params):
    type = type or base_type
    params = {**base_params, **params}
    return type(init_arg, **params)


def filter_dict(dct, keys):
    return {k: dct[k] for k in keys if k in dct}


@torch.no_grad()
def get_inception_statistics(X: torch.Tensor, batch_size=None, verbose=False):
    if batch_size is None:
        batch_size = X.size(0)
    model = InceptionV3().to(X.device)
    vectors = tqdm(X.split(batch_size), disable=not verbose)
    outputs = torch.cat([model(batch)[0] for batch in vectors]) \
        .flatten(1).cpu().numpy()
    return np.mean(outputs, axis=0), np.cov(outputs, rowvar=False)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
