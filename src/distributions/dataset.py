import torch

from .base import CompositeDistribution, _to_tensor


# TODO: Implement torchvision datasets


__all__ = ["TensorDatasetDistribution"]


class TensorDatasetDistribution(CompositeDistribution):
    def __init__(self, features, labels, device, store_on_device=True):
        self.store_on_device = store_on_device
        self.features = _to_tensor(features, None)
        self.labels = _to_tensor(labels, None)
        super().__init__(self.features.shape[1:], self.labels.unique(), device)
        self.to(self.device)

    def to(self, device):
        super().to(device)
        if self.store_on_device:
            self.features.to(device)
            self.labels.to(device)

    def sample(self, sample_shape=torch.Size(), *, return_labels=False):
        sample_shape = torch.Size(sample_shape)
        indices = torch.randint(0, self.features.size(0), sample_shape)

        if return_labels:
            return (self.features[indices].to(self.device),
                    self.labels[indices].to(self.device))
        return self.features[indices].to(self.device)

    def sample_components(self, sample_shape=torch.Size(), from_components=None):
        sample_shape = torch.Size(sample_shape)
        if from_components is None:
            from_components = self.component_labels.tolist()
        elif not from_components:
            return self.sample(sample_shape)

        indices = []
        for label in from_components:
            component = torch.where(self.labels == label)[0]
            indices.append(component[torch.randint(0, component.size(0),
                                                   sample_shape)])
        indices = torch.stack(indices)

        return self.features[indices].to(self.device)
