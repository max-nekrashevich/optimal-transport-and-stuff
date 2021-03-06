from matplotlib import projections
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ipywidgets import Output
from IPython.display import clear_output, display

from .utils import get_projection


def plot_heatmap(heatmap, xlim, ylim, **imshow_kwargs):
    yticks, xticks = heatmap.shape
    dx = .5 * (xlim[1] - xlim[0]) / (xticks - 1)
    dy = .5 * (ylim[1] - ylim[0]) / (yticks - 1)
    extent = (xlim[0] - dx, xlim[1] + dx, ylim[0] - dy, ylim[1] + dy)
    plt.imshow(heatmap[::-1, :], extent=extent, **imshow_kwargs)


@torch.no_grad()
def plot_samples(samples: torch.Tensor, **scatter_kwargs):
    _, dim = samples.size()
    ax = plt.gca()
    if ax.name != get_projection(dim):
        ax = plt.subplot(projection=get_projection(dim))

    ax.scatter(*samples.cpu().numpy().T, **scatter_kwargs)


@torch.no_grad()
def plot_arrows(samples, moved_samples):
    x, y = samples.cpu().numpy().T
    dx, dy = (moved_samples - samples).cpu().numpy().T
    plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1., width=.0025)


class Plotter:
    def __init__(self, source_dim, target_dim, show_progress=True) -> None:
        self.show = show_progress
        self.source_dim = source_dim
        self.target_dim = target_dim

    def init_widget(self):
        if self.show:
            self._plot_widget = Output()
            display(self._plot_widget)

    def update_widget(self, x, y, h_x, critic):
        figure = self.plot_step(x, y, h_x, critic)
        if self.show:
            with self._plot_widget:
                try:
                    clear_output(wait=True)
                    plt.show(block=False)
                    interrupted = False
                except KeyboardInterrupt:
                    self.close_widget()
                    interrupted = True

            if interrupted:
                raise KeyboardInterrupt
        else:
            plt.close()

        return figure

    def close_widget(self):
        if self.show:
            self._plot_widget.close()

    def plot_pdfs(self, source, target):
        raise NotImplementedError

    def plot_step(self, x, y, h_x, critic):
        raise NotImplementedError


class SimplePlotter(Plotter):
    def __init__(self, source_dim, target_dim, show_progress=True, pdf_params={}, step_params={}):
        super().__init__(source_dim, target_dim, show_progress)
        self.pdf_params = dict(
            figsize=(9,4),
            source_lims=None,
            source_kind="samples",
            target_lims=None,
            target_kind="samples",
            n_points=50,
            colorbar=True,
        )
        self.pdf_params.update(pdf_params)
        self.step_params = dict(
            figsize=(9,7),
            plot_x=True,
            plot_y=True,
            plot_h_x=True,
            x_color="blue",
            y_color="green",
            h_x_color="purple",
            dots_alpha=.5,
            heatmap_alpha=.5,
            n_arrows=128,
            colormap=cm.PRGn,
            n_points=50,
        )
        self.step_params.update(step_params)

    def _plot_density(self, distribution, kind, lims):
        if kind == "pdf":
            mesh = get_mesh(*lims, self.pdf_params["n_points"])
            heatmap = get_prob_heatmap(distribution, mesh).numpy()
            plot_heatmap(heatmap, *lims)
            if self.pdf_params["colorbar"]:
                plt.colorbar(label="Density")
        elif kind == "samples":
            samples = distribution.sample((self.pdf_params["n_points"],))
            plot_samples(samples)

    def plot_pdfs(self, source, target):
        plot_s = self.pdf_params["source_lims"] is not None
        plot_t = self.pdf_params["target_lims"] is not None

        figure = None
        if plot_s or plot_t:
            figure = plt.figure(figsize=self.pdf_params["figsize"])

        if plot_s:
            plt.subplot(1, 1 + plot_t, 1, title="Source PDF",
                        projection=get_projection(self.source_dim))
            self._plot_density(source,
                               self.pdf_params["source_kind"],
                               self.pdf_params["source_lims"])

        if plot_t:
            plt.subplot(1, 1 + plot_s, 1 + plot_s, title="Target PDF",
                        projection=get_projection(self.target_dim))
            self._plot_density(target,
                               self.pdf_params["target_kind"],
                               self.pdf_params["target_lims"])

        plt.tight_layout()

        if self.show:
            plt.show()
        else:
            plt.close()
        return figure

    def plot_step(self, x, y, h_x, critic):
        figure = plt.figure(figsize=self.step_params["figsize"])
        if self.source_dim == self.target_dim:
            plot_samples(x, color=self.step_params["x_color"],
                         alpha=self.step_params["dots_alpha"],
                         label="Source samples")
        plot_samples(y, color=self.step_params["y_color"],
                     alpha=self.step_params["dots_alpha"],
                     label="Target samples")
        plot_samples(h_x, color=self.step_params["h_x_color"],
                     alpha=self.step_params["dots_alpha"],
                     label="Moved samples")
        if self.source_dim == self.target_dim:
            plot_arrows(x[:self.step_params["n_arrows"]],
                        h_x[:self.step_params["n_arrows"]])

        if self.target_dim == 2:
            lims = (plt.gca().get_xlim(), plt.gca().get_ylim())
            mesh = get_mesh(*lims, self.step_params["n_points"])
            heatmap = get_critic_heatmap(critic, mesh).numpy()
            plot_heatmap(heatmap, *lims, alpha=self.step_params["heatmap_alpha"],
                        colormap=self.step_params["colormap"])
            plt.colorbar(label="Critic output")

        plt.legend(loc="upper left")
        plt.tight_layout()
        return figure
