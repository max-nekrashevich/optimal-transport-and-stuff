import torch

from torch.optim import Adam
from tqdm.auto import trange



class OTSolver:
    def __init__(self, critic, mover, cost,
                 optimizer=Adam, optimizer_params=dict(lr=5e-5),
                 n_samples=256, n_inner_iter=15,
                 plotter=None, plot_interval=25,
                 logger=None, log_plot_interval=100,
                 progress_bar=True, device=None):
        self.critic = critic.to(device)
        self.mover = mover.to(device)
        self.cost = cost

        self._critic_opt = optimizer(critic.parameters(), **optimizer_params)
        self._mover_opt = optimizer(mover.parameters(), **optimizer_params)

        self.plotter = plotter
        self.logger = logger

        self.show_progress = progress_bar
        self.n_samples = n_samples
        self.n_inner_iter = n_inner_iter
        self.plot_interval = plot_interval
        self.log_plot_interval = log_plot_interval
        self.device = device

        self._global_step = 1

    def fit(self, source, target, n_iter):
        if self.plotter:
            figure = self.plotter.plot_pdfs(source, target)

            if self.logger and figure:
                self.logger.log("PDFs", figure, close=True)

            self.plotter.init_widget()

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

            with torch.no_grad():
                loss = self.critic(y).mean() + mover_loss

                if self.show_progress:
                    progress_widget.set_postfix({"loss": loss.item()})

                if self.logger:
                    self.logger.log("loss", loss.item(), self._global_step + step)
                    self.logger.log("cost", cost.item(), self._global_step + step)

            if self.plotter and step % self.plot_interval == 0:
                figure = self.plotter.update_widget(x, y, h_x, self.critic)

                if self.logger and step % self.log_plot_interval == 0:
                    self.logger.log("Transport/Progress", figure,
                                    self._global_step + step, close=True)

        if self.plotter:
            self.plotter.close_widget()
            figure = self.plotter.plot_transport(x, y, h_x, self.critic)

            if self.logger:
                self.logger.log("Transport/Final", figure,
                                self._global_step + n_iter, close=True)

        self._global_step += n_iter
