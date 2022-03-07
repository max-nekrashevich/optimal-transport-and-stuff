import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch

from .utils import initializer


def get_mesh(xrange, yrange=None):
    yrange = yrange or xrange
    return torch.meshgrid(torch.linspace(*xrange),
                          torch.linspace(*yrange), indexing="xy")


def get_prob_heatmap(distribution, mesh):
    return distribution.log_prob(torch.dstack(mesh)).exp_()


@torch.no_grad()
def get_critic_heatmap(critic, mesh):
    device = next(critic.parameters()).device
    return critic(torch.dstack(mesh).to(device)).cpu().squeeze(-1)


def get_projection(dim):
    return "3d" if dim == 3 else "rectilinear"


def plot_heatmap(hmap, xlim, ylim=None, **imshow_kwargs):
    yticks, xticks = hmap.shape
    ylim = ylim or xlim
    dx = .5 * (xlim[1] - xlim[0]) / (xticks - 1)
    dy = .5 * (ylim[1] - ylim[0]) / (yticks - 1)
    extent = (xlim[0] - dx, xlim[1] + dx, ylim[0] - dy, ylim[1] + dy)
    plt.imshow(hmap[::-1, :], extent=extent, **imshow_kwargs)


@torch.no_grad()
def plot_samples(samples: torch.Tensor, **scatter_kwargs):
    _, dim = samples.size()
    ax = plt.gca()
    if ax.name != get_projection(dim):
        ax = plt.subplot(projection=get_projection(dim))

    ax.scatter(*samples.cpu().numpy().T, **scatter_kwargs)


@torch.no_grad()
def plot_arrows(samples, moved_samples):
    origins = samples.cpu().numpy().T
    lengths = (moved_samples - samples).cpu().numpy().T
    if len(origins) == 2:
        plt.quiver(*origins, *lengths, angles='xy',
                   scale_units='xy', scale=1., width=.0025)
    else:
        plt.quiver(*origins, *lengths, color="black")


def _plot_density(distribution, kind, lims, n_samples):
    if kind == "pdf":
        mesh = get_mesh(*lims)
        hmap = get_prob_heatmap(distribution, mesh).numpy()
        plot_heatmap(hmap, *lims)
        plt.colorbar(label="Density")
    elif kind == "samples":
        samples = distribution.sample((n_samples,))
        plot_samples(samples)


def plot_pdfs(source, target, source_dim, target_dim, *,
             figsize=(9, 4),
             source_kind="samples",
             source_lims=None,
             target_kind="samples",
             target_lims=None,
             n_samples=50):
    plot_s = source_kind is not None
    plot_t = target_kind is not None

    figure = None
    if plot_s or plot_t:
        figure = plt.figure(figsize=figsize)

    if plot_s:
        plt.subplot(1, 1 + plot_t, 1, title="Source Distribution",
                    projection=get_projection(source_dim))
        _plot_density(source, source_kind, source_lims, n_samples)

    if plot_t:
        plt.subplot(1, 1 + plot_s, 1 + plot_s, title="Target Distribution",
                    projection=get_projection(target_dim))
        _plot_density(target, target_kind, target_lims, n_samples)

    plt.tight_layout()
    return figure


def plot_transport(x, y, h_x, critic, *,
                   figsize=(9, 7),
                   x_color="blue",
                   y_color="green",
                   h_x_color="purple",
                   dots_alpha=.5,
                   n_arrows=128,
                   cmap=cm.PRGn,
                   hmap_alpha=.5):
    figure = plt.figure(figsize=figsize)
    source_dim = x.size(1)
    target_dim = y.size(1)

    if source_dim == target_dim:
        plot_samples(x, color=x_color,
                        alpha=dots_alpha, label="Source samples")

    plot_samples(y, color=y_color,
                    alpha=dots_alpha, label="Target samples")
    plot_samples(h_x, color=h_x_color,
                    alpha=dots_alpha, label="Moved samples")

    if source_dim == target_dim:
        plot_arrows(x[:n_arrows], h_x[:n_arrows])

    if target_dim == 2:
        lims = (plt.gca().get_xlim(), plt.gca().get_ylim())
        mesh = get_mesh(*lims)
        hmap = get_critic_heatmap(critic, mesh).numpy()
        plot_heatmap(hmap, *lims, alpha=hmap_alpha,
                    cmap=cmap)
        plt.colorbar(label="Critic score")

    plt.legend(loc="best")
    plt.tight_layout()
    return figure


class Plotter:
    def __init__(self, source_dim, target_dim):
        self.source_dim = source_dim
        self.target_dim = target_dim

    def plot_start(self, source, target):
        raise NotImplementedError

    def plot_step(self, x, y, h_x, **kwargs):
        raise NotImplementedError

    def plot_end(self, source, target, critic, mover):
        raise NotImplementedError


class SyntheticPlotter(Plotter):
    @initializer
    def __init__(self, source_dim, target_dim,
                 pdf_figsize=(9, 4),
                 pdf_kind_source="samples",
                 pdf_lims_source=None,
                 pdf_kind_target="samples",
                 pdf_lims_target=None,
                 pdf_n_samples=50,
                 transport_figsize=(9, 7),
                 transport_x_color="blue",
                 transport_y_color="green",
                 transport_h_x_color="purple",
                 transport_dots_alpha=.5,
                 transport_n_arrows=128,
                 transport_cmap=cm.PRGn,
                 transport_hmap_alpha=.5,
                 final_n_samples=512,
                 final_n_arrows=512):
        super().__init__(source_dim, target_dim)
        assert pdf_kind_source in (None, "pdf", "samples")
        assert pdf_kind_target in (None, "pdf", "samples")
        if pdf_lims_source and not isinstance(pdf_kind_source[0], tuple):
            self.pdf_lims_source = (pdf_lims_source,)
        if pdf_lims_target and not isinstance(pdf_kind_target[0], tuple):
            self.pdf_lims_target = (pdf_lims_target,)

    def plot_start(self, source, target):
        return plot_pdfs(source, target, self.source_dim, self.target_dim,
                         figsize=self.pdf_figsize, n_samples=self.pdf_n_samples,
                         source_kind=self.pdf_kind_source, source_lims=self.pdf_lims_source,
                         target_kind=self.pdf_kind_target, target_lims=self.pdf_lims_target,
                        )

    def plot_step(self, x, y, h_x, *, critic, **kwargs):
        return plot_transport(x, y, h_x, critic,
                              figsize=self.transport_figsize,
                              x_color=self.transport_x_color,
                              y_color=self.transport_y_color,
                              h_x_color=self.transport_h_x_color,
                              dots_alpha=self.transport_dots_alpha,
                              n_arrows=self.transport_n_arrows,
                              cmap=self.transport_cmap,
                              hmap_alpha=self.transport_hmap_alpha)

    @torch.no_grad()
    def plot_end(self, source, target, critic, mover):
        x = source.sample((self.final_n_samples,))
        y = target.sample((self.final_n_samples,))
        h_x = mover(x)

        return plot_transport(x, y, h_x, critic,
                              figsize=self.transport_figsize,
                              x_color=self.transport_x_color,
                              y_color=self.transport_y_color,
                              h_x_color=self.transport_h_x_color,
                              dots_alpha=self.transport_dots_alpha,
                              n_arrows=self.final_n_arrows,
                              cmap=self.transport_cmap,
                              hmap_alpha=self.transport_hmap_alpha)