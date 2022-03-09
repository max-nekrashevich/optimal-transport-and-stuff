from itertools import cycle

import matplotlib.cm as cm
import numpy as np
import torch

import matplotlib.pyplot as plt


def _get_mesh(xrange, yrange=None):
    yrange = yrange or xrange
    return torch.meshgrid(torch.linspace(*xrange),
                          torch.linspace(*yrange), indexing="xy")


@torch.no_grad()
def _get_critic_heatmap(critic, mesh):
    device = next(critic.parameters()).device
    return critic(torch.dstack(mesh).to(device)).cpu().squeeze(-1)


def _get_projection(dim):
    return "3d" if dim == (3,) else "rectilinear"


def _get_component_centers(data: torch.Tensor, labels: torch.Tensor):
    centers = []
    for label in labels.unique():
        centers.append(data[labels == label].mean(0))
    return torch.stack(centers)


def plot_heatmap(heatmap, xlim, ylim=None, **imshow_kwargs):
    yticks, xticks = heatmap.shape
    ylim = ylim or xlim
    dx = .5 * (xlim[1] - xlim[0]) / (xticks - 1)
    dy = .5 * (ylim[1] - ylim[0]) / (yticks - 1)
    extent = (xlim[0] - dx, xlim[1] + dx, ylim[0] - dy, ylim[1] + dy)
    plt.imshow(heatmap[::-1, :], extent=extent, **imshow_kwargs)


@torch.no_grad()
def plot_samples(samples: torch.Tensor, **scatter_kwargs):
    plt.gca().scatter(*samples.cpu().numpy().T, **scatter_kwargs)


@torch.no_grad()
def plot_quiver(from_, to_, **quiver_kwargs):
    origins = from_.cpu().numpy().T
    lengths = (to_ - from_).cpu().numpy().T
    if len(origins) == 2:
        plt.quiver(*origins, *lengths, angles='xy', scale_units='xy',
                   scale=1., width=.0025, **quiver_kwargs)
    else:
        plt.quiver(*origins, *lengths, color="black", **quiver_kwargs)


def close_figure(figure):
    plt.close(figure)


def show_figure():
    plt.show(block=False)


def plot_density(distribution, kind, lims, n_samples):
    if kind == "pdf":
        mesh = _get_mesh(*lims)
        heatmap = distribution.log_prob(torch.dstack(mesh)).exp_().numpy()
        plot_heatmap(heatmap, *lims)
        plt.colorbar(label="Density")
    elif kind == "samples":
        samples = distribution.sample((n_samples,))
        plot_samples(samples)


def plot_transport(x, y, h_x, labels, *, critic=None,
                   plot_source=True,
                   plot_target=True,
                   plot_critic=True,
                   plot_arrows=True,
                   legend=True,
                   show=True,
                   figsize=(9, 7),
                   aggregate_arrows=True,
                   n_arrows=128,
                   source_colors=None,
                   target_color="darkseagreen",
                   dots_alpha=.5,
                   colormap=None,
                   heatmap_alpha=.5):
    source_colors = source_colors or [f"C{i}" for i in range(10)]
    colormap = colormap or cm.PRGn
    figure = plt.figure(figsize=figsize)
    source_dim = x.shape[1:]
    target_dim = y.shape[1:]
    components = labels.unique()
    plt.subplot(projection=_get_projection(target_dim))

    if plot_source and source_dim == target_dim:
        for component, color in zip(components, cycle(source_colors)):
            plot_samples(x[labels == component], color=color,
                         alpha=dots_alpha, label=f"Source component {component}")

    if plot_target:
        plot_samples(y, color=target_color,
                     alpha=dots_alpha, label="Target")

    for component, color in zip(components, cycle(source_colors)):
        plot_samples(h_x[labels == component], color=color, marker="v",
                     alpha=dots_alpha, label=f"Moved component {component}")

    if plot_arrows and source_dim == target_dim:
        if aggregate_arrows:
            arrows_from = _get_component_centers(x, labels)
            arrows_to = _get_component_centers(h_x, labels)
        else:
            ix = torch.randint(0, x.size(0), (n_arrows,))
            arrows_from = x[ix]
            arrows_to = h_x[ix]
        plot_quiver(arrows_from, arrows_to)

    if plot_critic and target_dim == (2,):
        lims = (plt.gca().get_xlim(), plt.gca().get_ylim())
        mesh = _get_mesh(*lims)
        heatmap = _get_critic_heatmap(critic, mesh).numpy()
        plot_heatmap(heatmap, *lims, alpha=heatmap_alpha,
                    cmap=colormap)
        plt.colorbar(label="Critic score")

    if legend: plt.legend(loc="upper left")
    plt.tight_layout()
    if show: plt.show(block=False)
    return figure


# def plot_start(source, target, *,
#                source_kind="samples", target_kind="samples",
#                source_lims=None, target_lims=None,
#                figsize=(9, 4), n_samples=50):
#     plot_s = source_kind is not None
#     plot_t = target_kind is not None

#     figure = None
#     if plot_s or plot_t:
#         figure = plt.figure(figsize=figsize)

#     if plot_s:
#         plt.subplot(1, 1 + plot_t, 1, title="Source Distribution",
#                     projection=_get_projection(source.event_shape))
#         plot_density(source, source_kind, source_lims, n_samples)

#     if plot_t:
#         plt.subplot(1, 1 + plot_s, 1 + plot_s, title="Target Distribution",
#                     projection=_get_projection(target.event_shape))
#         plot_density(target, target_kind, target_lims, n_samples)

#     plt.tight_layout()
#     return figure
