import numpy as np
import torch.nn as nn

from ..utils import nwise


__all__ = ["mlp", "mnistnet_g", "mnistnet_d", "mnistnet_h"]


def mlp(input_size: int, output_size: int = 1, *,
        hidden_size: int = 32, num_layers: int = 4):
    layer_sizes = [hidden_size] * (num_layers - 1) + [output_size]
    modules: list[nn.Module] = [nn.Linear(input_size, layer_sizes[0])]

    for in_size, out_size in nwise(layer_sizes):
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(.1))
        modules.append(nn.Linear(in_size, out_size))

    return nn.Sequential(*modules)


def mnistnet_d(data_dim: tuple = (1, 28, 28)):
    n_features = int(np.prod(data_dim))
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 1)
    )
    return model


def mnistnet_g(data_dim: tuple, code_size: int):
    n_features = int(np.prod(data_dim))
    model = nn.Sequential(
        *block(code_size, 128, normalize=False),
        *block(128, 256),
        *block(256, 512),
        *block(512, 1024),
        nn.Linear(1024, n_features),
        nn.Unflatten(1, tuple(data_dim)),
        nn.Sigmoid()
    )
    return model


def mnistnet_h(data_dim: tuple = (1, 28, 28)):
    n_features = int(np.prod(data_dim))
    model = nn.Sequential(
        nn.Flatten(),
        *block(n_features, n_features, normalize=False),
        *block(n_features, n_features),
        *block(n_features, n_features),
        *block(n_features, n_features),
        *block(n_features, n_features),
        nn.Unflatten(1, tuple(data_dim)),
        nn.Sigmoid()
    )
    return model


def block(in_features, out_features, normalize=True):
    layers = []
    layers.append(nn.Linear(in_features, out_features))
    if normalize:
        layers.append(nn.BatchNorm1d(out_features, 0.8))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers
