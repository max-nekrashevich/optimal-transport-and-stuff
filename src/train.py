import torch
import torch.optim as o

from tqdm.auto import trange


def _log(logger, tag, data, **kwargs):
    if logger is not None and data:
        logger.log(tag, data, **kwargs)


def train(source, target, mover, critic, cost_func, *,
          n_iter, n_samples, n_iter_mover=10, n_iter_critic=1,
          l=.05,
          optimizer=o.Adam,
          optimizer_params=dict(lr=5e-5),
          logger=None,
          plotter=None,
          progress_bar=True):
    mover_optimizer = optimizer(mover.parameters(), **optimizer_params)
    critic_optimizer = optimizer(critic.parameters(), **optimizer_params)

    if plotter: plotter.init_widget()

    progress_widget = trange(n_iter, disable=not progress_bar)

    for _ in progress_widget:
        x, labels = source.sample((n_samples,), return_labels=True)
        y = target.sample((n_samples,))

        for _ in range(n_iter_mover):
            h_x = mover(x)
            mover_optimizer.zero_grad()
            cost = l * cost_func(x, h_x).mean()
            mover_loss = cost - critic(h_x).mean()
            mover_loss.backward()
            mover_optimizer.step()

        for _ in range(n_iter_critic):
            critic_optimizer.zero_grad()
            critic_loss = critic(h_x.detach()).mean() - critic(y).mean()
            critic_loss.backward()
            critic_optimizer.step()

        if plotter:
            figure = plotter.plot_step(x, y, h_x, labels, critic=critic)
            _log(logger, "Transport", figure, advance=False, close=True)

        with torch.no_grad():
            loss = critic(y).mean() + mover_loss

        progress_widget.set_postfix({"loss": loss.item()})

        _log(logger, "loss", loss.item(), advance=False)
        _log(logger, "cost", cost.item())

    if plotter:
        plotter.close_widget()
        figure = plotter.plot_end(source, target, mover, critic=critic)
        _log(logger, "Result", figure, advance=False, close=True)
