from torch.optim import Adam
from tqdm.auto import trange



class OTSolver:
    def __init__(self, critic, mover, cost_func,
                 optimizer=Adam, optimizer_params=dict(lr=5e-5, betas=(.0, .9)),
                 n_samples=256, n_inner_iter=25,
                 logger=None, plotter=None, plot_interval=25,
                 progress_bar=True, device=None):
        self.critic = critic.to(device)
        self.mover = mover.to(device)
        self.cost = cost_func

        self._critic_opt = optimizer(critic.parameters(), **optimizer_params)
        self._mover_opt = optimizer(mover.parameters(), **optimizer_params)

        self.plotter = plotter

        self.progress_bar = progress_bar
        self.n_samples = n_samples
        self.n_inner_iter = n_inner_iter
        self.plot_interval = plot_interval
        self.device = device

    def fit(self, source, target, n_iter):
        progress_widget = trange(n_iter, disable=not self.progress_bar)

        if self.plotter:
            fig = self.plotter.plot_pdfs(source, target)

            # if self.logger:

            self.plotter.init_widget()

        for step in progress_widget:
            x = source.sample((self.n_samples,)).to(self.device)
            y = target.sample((self.n_samples,)).to(self.device)

            for _ in range(self.n_inner_iter):
                h_x = self.mover(x)

                self._mover_opt.zero_grad()
                mover_loss = self.cost(x.detach(), h_x).mean() - self.critic(h_x).mean()
                mover_loss.backward()
                self._mover_opt.step()

            self._critic_opt.zero_grad()
            critic_loss = self.critic(h_x.detach()).mean() - self.critic(y).mean()
            critic_loss.backward()
            self._critic_opt.step()

            if self.plotter and step % self.plot_interval == 0:
                figure = self.plotter.update_widget(x, y, h_x, self.critic)

                # if self.logger:

        if self.plotter:
            self.plotter.close_widget()
            figure = self.plotter.plot_transport(x, y, h_x, self.critic)

            # if self.logger:
