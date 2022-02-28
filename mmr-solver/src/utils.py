import numpy as np

import torch

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
