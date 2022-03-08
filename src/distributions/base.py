from collections import Counter

import torch


# TODO: MultivariateNormal
# TODO: Implement log_probs
# TODO: Validate args


__all__ = ["Distribution",
           "CompositeDistribution",
           "Uniform",
           "Normal",
           "CurveDistribution",
           "DiscreteMixture",
           "GaussianMixture"]


def _to_tensor(data, device):
    if isinstance(data, torch.Tensor):
        return data.clone().detach().to(device)
    return torch.tensor(data, device=device)


class Distribution:
    def __init__(self, event_shape, device):
        self.event_shape = torch.Size(event_shape)
        self.device = device

    def to(self, device):
        raise NotImplementedError

    def sample(self, sample_shape=torch.Size(), **kwargs):
        raise NotImplementedError


class CompositeDistribution(Distribution):
    def __init__(self, event_shape, component_labels, device):
        self.component_labels = _to_tensor(component_labels, device)
        self._component_ix = {c.item(): ix for ix, c in enumerate(self.component_labels)}
        super().__init__(event_shape, device)

    def to(self, device):
        self.component_labels.to(device)

    def sample(self, sample_shape=torch.Size(), *, return_labels=False):
        raise NotImplementedError

    def sample_components(self, sample_shape, from_components=None):
        raise NotImplementedError


class Uniform(Distribution):
    def __init__(self, low, high, *, device=None):
        self.low = _to_tensor(low, device)
        self.high = _to_tensor(high, device)
        super().__init__(low.size(), device)

    def to(self, device):
        self.low.to(device)
        self.high.to(device)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), **kwargs):
        sample_shape = torch.Size(sample_shape)
        random = torch.rand(sample_shape + self.event_shape,
                            dtype=self.low.dtype, device=self.device)
        return self.low + random * (self.high - self.low)


class Normal(Distribution):
    def __init__(self, loc, scale, *, device=None):
        self.loc = _to_tensor(loc, device)
        self.scale = _to_tensor(scale, device)
        super().__init__(loc.size(), device)

    def to(self, device):
        self.loc.to(device)
        self.scale.to(device)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), **kwargs):
        sample_shape = torch.Size(sample_shape)
        random = torch.randn(sample_shape + self.event_shape,
                             dtype=self.loc.dtype, device=self.device)
        return self.loc + random * self.scale


class CurveDistribution(Distribution):
    def __init__(self, curve, *, device=None):
        self.curve = curve
        self.t_distribution = Uniform(0., 1., device=device)
        super().__init__(curve(torch.tensor(0.)).size(), device)

    def sample(self, sample_shape=torch.Size(), **kwargs):
        t_samples = self.t_distribution.sample(sample_shape)
        return self.curve(t_samples)


class DiscreteMixture(CompositeDistribution):
    def __init__(self, components, probs, labels, *, device=None):
        self.components = components
        for component in self.components:
            component.to(device)
        self.probs = _to_tensor(probs, device)
        event_shape = components[0].event_shape
        super().__init__(event_shape, labels, device)

    def to(self, device):
        super().to(device)
        for component in self.components:
            component.to(device)
        self.probs.to(device)
        self.components.to(device)

    def sample(self, sample_shape=torch.Size(), *, return_labels=False):
        sample_shape = torch.Size(sample_shape)

        indices = torch.multinomial(self.probs, sample_shape.numel(), replacement=True)
        samples = torch.empty(sample_shape.numel(), *self.event_shape, device=self.device)
        for index, count in Counter(indices).items():
            samples[indices == index] = self.components[index].sample((count,))
        samples = samples.view(sample_shape + self.event_shape)
        labels = self.component_labels[indices].view(sample_shape)

        if return_labels:
            return samples, labels
        return samples

    def sample_components(self, sample_shape=torch.Size(), from_components=None):
        sample_shape = torch.Size(sample_shape)
        if from_components is None:
            from_components = self.component_labels.tolist()
        elif not from_components:
            return self.sample(sample_shape)
        samples = []
        for label in from_components:
            samples.append(self.components[self._component_ix[label]].sample(sample_shape))
        samples = torch.stack(samples)
        return samples


class GaussianMixture(DiscreteMixture):
    def __init__(self, locs, scales, *, probs=None, device=None):
        components = [Normal(loc, scale, device=device) for loc, scale in zip(locs, scales)]
        labels = torch.arange(0, locs.size(0), )
        if probs is None: probs = torch.ones_like(labels) / locs.size(0)
        super().__init__(components, probs, labels, device=device)
