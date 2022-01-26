import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ipywidgets import Output
from IPython.display import clear_output, display

# def get_prob(distribution):
#     def get(tensor):
#         return distribution.log_prob(tensor).exp_()
#     return get


# def apply_critic(critic):
#     DEVICE = next(critic.parameters()).device
#     @torch.no_grad()
#     def apply(tensor):
#         return critic(tensor.to(DEVICE))
#     return apply


# def plot_grid(func, xlim=None, ylim=None, n_pixels=50,
#               colorbar=False, colorbar_label=None, **imshow_kwargs):
#     ylim = ylim or xlim or plt.gca().get_ylim()
#     xlim = xlim or plt.gca().get_xlim()
#     mesh = torch.meshgrid(torch.linspace(*xlim, n_pixels),
#                           torch.linspace(*ylim, n_pixels), indexing="xy")
#     grid = func(torch.dstack(mesh)).cpu().numpy()
#     dx = .5 * (xlim[1] - xlim[0]) / (n_pixels - 1)
#     dy = .5 * (ylim[1] - ylim[0]) / (n_pixels - 1)
#     extent = (xlim[0] - dx, xlim[1] + dx, ylim[0] - dy, ylim[1] + dy)
#     plt.imshow(grid[::-1, :], extent=extent, **imshow_kwargs)
#     if colorbar:
#         plt.colorbar(label=colorbar_label)


def get_mesh(xlim, ylim, n_points):
    if not isinstance(n_points, tuple):
        n_points = (n_points, n_points)
    return torch.meshgrid(torch.linspace(*xlim, n_points[0]),
                          torch.linspace(*ylim, n_points[1]), indexing="xy")


def get_prob_heatmap(distribution, mesh):
    return distribution.log_prob(torch.dstack(mesh)).exp_()


@torch.no_grad()
def get_critic_heatmap(critic, mesh):
    device = next(critic.parameters()).device
    return critic(torch.dstack(mesh).to(device)).cpu().squeeze(-1)


def plot_heatmap(hmap, xlim, ylim, **imshow_kwargs):
    yticks, xticks = hmap.shape
    dx = .5 * (xlim[1] - xlim[0]) / (xticks - 1)
    dy = .5 * (ylim[1] - ylim[0]) / (yticks - 1)
    extent = (xlim[0] - dx, xlim[1] + dx, ylim[0] - dy, ylim[1] + dy)
    plt.imshow(hmap[::-1, :], extent=extent, **imshow_kwargs)


@torch.no_grad()
def plot_arrows(samples, moved_samples):
    x, y = samples.cpu().numpy().T
    dx, dy = (moved_samples - samples).cpu().numpy().T
    plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1., width=.0025)


class Plotter:
    def __init__(self, source_lims, target_lims):
        self.pdf_params = dict(
            figsize=(12,5),
            plot_source=True,
            plot_target=True,
            source_lims=source_lims,
            target_lims=target_lims,
            n_points=50,
            colorbar=True,
        )
        self.transport_params = dict(
            figsize=(12,10),
            plot_x=True,
            plot_y=True,
            plot_h_x=True,
            x_color="blue",
            y_color="green",
            h_x_color="purple",
            dots_alpha=.5,
            hmap_alpha=.5,
            n_arrows=128,
            cmap=cm.PRGn,
            n_points=50,
        )

    def init_widget(self):
        self._plot_widget = Output()
        display(self._plot_widget)

    def update_widget(self, x, y, h_x, critic):
        with self._plot_widget:
            clear_output(wait=True)
            figure = self.plot_transport(x, y, h_x, critic)
            plt.show(block=False)
        return figure

    def close_widget(self):
        self._plot_widget.close()

    def _plot_density(self, distribution, lims):
        mesh = get_mesh(*lims, self.pdf_params["n_points"])
        hmap = get_prob_heatmap(distribution, mesh).numpy()
        plot_heatmap(hmap, *lims)
        if self.pdf_params["colorbar"]:
            plt.colorbar(label="Density")

    def plot_pdfs(self, source, target):
        plot_s = self.pdf_params["plot_source"]
        plot_t = self.pdf_params["plot_target"]

        figure = None
        if plot_s or plot_t:
            figure = plt.figure(figsize=self.pdf_params["figsize"])

        if plot_s:
            plt.subplot(1, 1 + plot_t, 1, title="Source PDF")
            self._plot_density(source, self.pdf_params["source_lims"])

        if plot_t:
            plt.subplot(1, 1 + plot_s, 1 + plot_s, title="Target PDF")
            self._plot_density(target, self.pdf_params["target_lims"])

        plt.show()
        return figure

    def plot_transport(self, x, y, h_x, critic):
        figure = plt.figure(figsize=self.transport_params["figsize"])
        plt.scatter(*x.detach().cpu().numpy().T,
                    color=self.transport_params["x_color"], label="Source samples",
                    alpha=self.transport_params["dots_alpha"])
        plt.scatter(*y.detach().cpu().numpy().T,
                    color=self.transport_params["y_color"], label="Target samples",
                    alpha=self.transport_params["dots_alpha"])
        plt.scatter(*h_x.detach().cpu().numpy().T,
                    color=self.transport_params["h_x_color"], label="Moved samples",
                    alpha=self.transport_params["dots_alpha"])
        plot_arrows(x[:self.transport_params["n_arrows"]],
                    h_x[:self.transport_params["n_arrows"]])

        lims = (plt.gca().get_xlim(), plt.gca().get_ylim())
        mesh = get_mesh(*lims, self.transport_params["n_points"])
        hmap = get_critic_heatmap(critic, mesh).numpy()
        plot_heatmap(hmap, *lims, alpha=self.transport_params["hmap_alpha"],
                     cmap=self.transport_params["cmap"])
        plt.colorbar(label="Critic output")

        plt.legend(loc="upper left")
        return figure
