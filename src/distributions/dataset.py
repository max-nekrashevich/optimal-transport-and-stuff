import torch

from .base import CompositeDistribution, _to_tensor


# TODO: Implement torchvision datasets


__all__ = ["TensorDatasetDistribution"]


class TensorDatasetDistribution(CompositeDistribution):
    def __init__(self, features, labels, device):
        self.features = _to_tensor(features, device)
        self.labels = _to_tensor(labels, device)
        super().__init__(self.features.shape[1:], self.labels.unique(), device)

    def to(self, device):
        super().to(device)
        self.features.to(device)
        self.labels.to(device)

    def sample(self, sample_shape=torch.Size(), *, return_labels=False):
        sample_shape = torch.Size(sample_shape)
        indices = torch.randint(0, self.features.size(0), sample_shape)

        if return_labels:
            return self.features[indices], self.labels[indices]
        return self.features[indices]

    def sample_components(self, sample_shape=torch.Size(), from_components=()):
        sample_shape = torch.Size(sample_shape)
        if from_components is None:
            from_components = self.component_labels.tolist()
        elif not from_components:
            return self.sample(sample_shape)

        indices = []
        for label in from_components:
            component = torch.where(self.labels == label)[0]
            indices.append(component[torch.randint(0, component.size(0), sample_shape)])
        indices = torch.stack(indices)

        return self.features[indices]
