import torch

from ipywidgets import Output
from IPython.display import clear_output, display

from . import matplotlib as mpl
from . import plotly as ply


# TODO: Image plotter
# TODO: Rework end to generate subplots for component


__all__ = ["Plotter", "ImagePlotter"]


_backends = {"matplotlib": mpl,
             "plotly": ply}


class Plotter:
    def __init__(self, backend="matplotlib", *,
                 n_samples=50, plot_interval=25, show=True,
                 **plot_params):
        self._backend = _backends[backend]
        self.plot_interval = plot_interval
        self.show = show
        self.n_samples = n_samples
        self.plot_params = plot_params
        self._step_num = -1

    def init_widget(self):
        self._widget = Output()
        if self.show:
            display(self._widget)

    def close_widget(self):
        self._widget.close()

    def _get_step_figure(self, x, y, h_x, labels, *, critic=None):
        return self._backend.get_transport_figure(x, y, h_x, labels,
                                                       critic=critic,
                                                       **self.plot_params)

    def plot_step(self, x, y, h_x, labels, *, critic=None):
        self._step_num += 1
        if self._step_num % self.plot_interval != 0:
            return None
        with self._widget:
            try:
                clear_output(wait=True)
                figure = self._get_step_figure(x, y, h_x, labels, critic=critic)
                interrupted = False
            except KeyboardInterrupt:
                self._widget.close()
                interrupted = True

        if interrupted:
            raise KeyboardInterrupt
        return figure

    @torch.no_grad()
    def plot_end(self, source, target, mover, *, critic=None):
        x = source.sample_components((self.n_samples,),
                                     from_components=None).flatten(end_dim=1)
        labels = source.component_labels.repeat_interleave(self.n_samples)
        y = target.sample((self.n_samples,))
        h_x = mover(x)
        return self._backend._get_step_figure(x, y, h_x, labels,
                                              critic=critic,
                                              **self.plot_params)


class ImagePlotter(Plotter):
    def _get_step_figure(self, x, y, h_x, labels, *, critic=None):
        return self._backend.get_images_figure(x, y, h_x, labels,
                                                       critic=critic,
                                                       **self.plot_params)
