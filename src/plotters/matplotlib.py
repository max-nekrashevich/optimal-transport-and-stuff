from itertools import cycle

import matplotlib.cm as cm
import numpy as np
import torch

import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_pil_image


def _get_mesh(xrange, yrange=None, n_steps=100):
    yrange = yrange or xrange
    return torch.meshgrid(torch.linspace(*xrange, n_steps),
                          torch.linspace(*yrange, n_steps), indexing="xy")


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


def _get_grid_dims(n_plots):
    for n_rows in range(int(np.sqrt(n_plots)), 1, -1):
        if n_plots % n_rows == 0:
            return n_rows, n_plots // n_rows
    return 1, n_plots


def plot_heatmap(heatmap, xlim, ylim=None, *, ax=None, **imshow_kwargs):
    if ax is None: ax = plt.gca()
    yticks, xticks = heatmap.shape
    ylim = ylim or xlim
    dx = .5 * (xlim[1] - xlim[0]) / (xticks - 1)
    dy = .5 * (ylim[1] - ylim[0]) / (yticks - 1)
    extent = (xlim[0] - dx, xlim[1] + dx, ylim[0] - dy, ylim[1] + dy)
    ax.imshow(heatmap[::-1, :], extent=extent, **imshow_kwargs)


@torch.no_grad()
def plot_samples(samples: torch.Tensor, ax=None, **scatter_kwargs):
    if ax is None: ax = plt.gca()
    ax.scatter(*samples.cpu().numpy().T, **scatter_kwargs)


@torch.no_grad()
def show_image(image: torch.Tensor, ax=None, **imshow_kwargs):
    if ax is None: ax = plt.gca()
    ax.imshow(to_pil_image(image.cpu().clamp(0, 1)), **imshow_kwargs)


@torch.no_grad()
def plot_quiver(from_, to_, *, ax=None, **quiver_kwargs):
    if ax is None: ax = plt.gca()
    origins = from_.cpu().numpy().T
    lengths = (to_ - from_).cpu().numpy().T
    if len(origins) == 2:
        ax.quiver(*origins, *lengths, angles='xy', scale_units='xy',
                  scale=1., width=.0025, **quiver_kwargs)
    else:
        ax.quiver(*origins, *lengths, color="black", **quiver_kwargs)


def close_figure(figure):
    plt.close(figure)


def show_figure():
    plt.show(block=False)


def plot_density(distribution, kind, lims, n_samples, *, ax=None):
    if kind == "pdf":
        mesh = _get_mesh(*lims)
        heatmap = distribution.log_prob(torch.dstack(mesh)).exp_().numpy()
        plot_heatmap(heatmap, *lims, ax=ax)
        plt.colorbar(label="Density")
    elif kind == "samples":
        samples = distribution.sample((n_samples,))
        plot_samples(samples, ax=ax)


def plot_transport(x, y, h_x, labels, *, critic=None, ax=None,
                   plot_source=True,
                   plot_target=True,
                   plot_critic=True,
                   plot_arrows=True,
                   legend=True,
                   aggregate_arrows=True,
                   n_arrows=128,
                   source_colors=None,
                   target_color="darkseagreen",
                   dots_alpha=.5,
                   colormap=None,
                   heatmap_alpha=.5):
    source_colors = source_colors or [f"C{i}" for i in range(10)]
    colormap = colormap or cm.PRGn
    source_dim = x.shape[1:]
    target_dim = y.shape[1:]
    components = labels.unique()
    if ax is None: ax = plt.gca()

    if plot_source and source_dim == target_dim:
        for component, color in zip(components, cycle(source_colors)):
            plot_samples(x[labels == component], ax=ax, color=color,
                         alpha=dots_alpha, label=f"Source component {component}")

    if plot_target:
        plot_samples(y, color=target_color,
                     alpha=dots_alpha, label="Target")

    for component, color in zip(components, cycle(source_colors)):
        plot_samples(h_x[labels == component], ax=ax, color=color, marker="v",
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
        # plt.colorbar(label="Critic score")

    if legend: ax.legend(loc="best")


def get_transport_figure(x, y, h_x, labels, *, critic=None,
                         figsize=(9, 7),
                         show=True,
                         **plot_transport_params):
    figure = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=_get_projection(y.shape[1:]))
    plot_transport(x, y, h_x, labels, critic=critic, ax=ax, **plot_transport_params)
    plt.tight_layout()
    if show: plt.show(block=False)
    return figure


def get_images_figure(x, y, h_x, labels, *, critic, n_images,
                      plot_source=True,
                      img_scale=16,
                      show=True,
                      **imshow_kwargs):
    n_rows, n_cols = _get_grid_dims(n_images)
    if plot_source: n_rows *= 2
    img_h, img_w = h_x.shape[-2:]
    figsize = (n_cols * img_w / img_scale, n_rows * img_h / img_scale)
    figure, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=True)
    if plot_source:
        for image, ax, label in zip(x, axes[::2].ravel(), labels):
            show_image(image, ax, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_title(f"Component {label}")
        for image, ax, label in zip(h_x, axes[1::2].ravel(), labels):
            show_image(image, ax, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            # ax.set_title(f"Moved component {label}")
    else:
        for image, ax, label in zip(h_x, axes.ravel(), labels):
            show_image(image, ax, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_title(f"Component {label}")
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
