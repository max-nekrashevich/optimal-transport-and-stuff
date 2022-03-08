import sklearn.datasets as datasets
import torch

from .base import CompositeDistribution, _to_tensor


# TODO: Implement other datasets


__all__ = ["MoonsDistribution"]


class MoonsDistribution(CompositeDistribution):
    def __init__(self, noise=.1, *, device=None):
        self.noise = noise
        super().__init__((2,), (0, 1), device)

    def sample(self, sample_shape, *, return_labels=False):
        sample_shape = torch.Size(sample_shape)
        samples, labels = datasets.make_moons(sample_shape.numel(), noise=self.noise)
        samples = torch.tensor(samples, device=self.device).view(sample_shape + self.event_shape)
        labels = torch.tensor(labels, device=self.device).view(sample_shape)

        if return_labels:
            return samples, labels
        return samples

    def sample_components(self, sample_shape=torch.Size(), from_components=()):
        sample_shape = torch.Size(sample_shape)
        if from_components is None:
            from_components = self.component_labels.tolist()
        elif not from_components:
            return self.sample(sample_shape)
        n_samples = torch.zeros_like(self.component_labels)
        n_samples[_to_tensor(from_components, self.device)] = sample_shape.numel()
        samples, _ = datasets.make_moons(n_samples.numpy(), noise=self.noise)
        return samples
