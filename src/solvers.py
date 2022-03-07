import matplotlib.pyplot as plt
import torch

from ipywidgets import Output
from IPython.display import clear_output, display
from torch.optim import Adam
from tqdm.auto import trange


class OTSolver:
    def __init__(self, critic, mover, cost,
                 optimizer=Adam, optimizer_params=dict(lr=5e-5),
                 n_samples=256, n_inner_iter=15,
                 plotter=None, plot_interval=25,
                 logger=None, log_plot_interval=100,
                 progress_bar=True, widget=True, device=None):
        self.critic = critic.to(device)
        self.mover = mover.to(device)
        self.cost = cost

        self._critic_opt = optimizer(critic.parameters(), **optimizer_params)
        self._mover_opt = optimizer(mover.parameters(), **optimizer_params)

        self.plotter = plotter
        self.logger = logger

        self.show_progress = progress_bar
        self.widget = widget
        self.n_samples = n_samples
        self.n_inner_iter = n_inner_iter
        self.plot_interval = plot_interval
        self.log_plot_interval = log_plot_interval
        self.device = device

    def fit(self, source, target, n_iter):
        if self.plotter:
            figure = self.plotter.plot_start(source, target)
            plt.show(block=False)

            if self.logger and figure:
                self.logger.log("PDFs", figure, advance=False, close=True)

            if self.widget:
                plot_widget = Output()
                display(plot_widget)

        progress_widget = trange(n_iter, disable=not self.show_progress)

        for step in progress_widget:
            x = source.sample((self.n_samples,)).to(self.device)
            y = target.sample((self.n_samples,)).to(self.device)

            for _ in range(self.n_inner_iter):
                h_x = self.mover(x)

                self._mover_opt.zero_grad()
                cost = self.cost(x, h_x).mean()
                mover_loss = cost - self.critic(h_x).mean()
                mover_loss.backward()
                self._mover_opt.step()

            self._critic_opt.zero_grad()
            critic_loss = self.critic(h_x.detach()).mean() - self.critic(y).mean()
            critic_loss.backward()
            self._critic_opt.step()

            if self.plotter and step % self.plot_interval == 0:

                figure = self.plotter.plot_step(x, y, h_x, critic=self.critic)
                if self.widget:
                    with plot_widget:
                        try:
                            clear_output(wait=True)
                            plt.show(block=False)
                            interrupted = False
                        except KeyboardInterrupt:
                            plot_widget.close()
                            interrupted = True

                    if interrupted:
                        raise KeyboardInterrupt
                else:
                    plt.close()

                if self.logger and step % self.log_plot_interval == 0:
                    self.logger.log("Transport/Progress", figure,
                                    advance=False, close=True)

            with torch.no_grad():
                loss = self.critic(y).mean() + mover_loss

                if self.show_progress:
                    progress_widget.set_postfix({"loss": loss.item()})

                if self.logger:
                    self.logger.log("loss", loss.item(), advance=False)
                    self.logger.log("cost", cost.item())

        if self.widget:
            plot_widget.close()

        if self.plotter:
            figure = self.plotter.plot_end(source, target, self.critic, self.mover)
            plt.show(block=False)

            if self.logger:
                self.logger.log("Transport/Final", figure, advance=False, close=True)
