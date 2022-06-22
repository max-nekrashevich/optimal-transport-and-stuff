import torch
import torch.nn as nn
import torch.optim as o

from tqdm.auto import trange

from .costs import Cost
from .distributions import BasicDistribution, CompositeDistribution


def _log(logger, tag, data, **kwargs):
    if logger is not None and data:
        logger.log(tag, data, **kwargs)


def train(source: CompositeDistribution, target: BasicDistribution,
          mover: nn.Module, critic: nn.Module, cost: Cost, *,
          n_iter: int, n_samples: int,
          n_iter_mover: int = 10, n_iter_critic: int = 1, n_iter_cost: int = 0,
          alpha: float = .05,
          optimizer=o.Adam,
          optimizer_params=dict(lr=5e-5),
          scheduler=None,
          scheduler_params=dict(),
          logger=None,
          plotter=None,
          progress_bar=True):
    mover_optimizer = optimizer(mover.parameters(), **optimizer_params)
    critic_optimizer = optimizer(critic.parameters(), **optimizer_params)
    if n_iter_cost:
        cost_optimizer = optimizer(cost.parameters(), **optimizer_params)

    if scheduler is not None:
        mover_scheduler = scheduler(mover_optimizer, **scheduler_params)
        critic_scheduler = scheduler(critic_optimizer, **scheduler_params)
    if scheduler is not None and n_iter_cost:
        cost_scheduler = scheduler(cost_optimizer, **scheduler_params)

    if plotter: plotter.init_widget()

    progress_widget = trange(n_iter, disable=not progress_bar)

    for _ in progress_widget:
        x, labels = source.sample((n_samples,), return_labels=True)
        y = target.sample((n_samples,))

        for _ in range(n_iter_cost):
            h_x = mover(x)
            cost_optimizer.zero_grad()
            cost_val = alpha * cost(x, h_x).mean()
            cost_val.backward()
            cost_optimizer.step()

        if scheduler is not None and n_iter_cost:
            cost_scheduler.step()

        for _ in range(n_iter_mover):
            h_x = mover(x)
            mover_optimizer.zero_grad()
            cost_val = alpha * cost(x, h_x).mean()
            mover_loss = cost_val - critic(h_x).mean()
            mover_loss.backward()
            mover_optimizer.step()

        if scheduler is not None:
            mover_scheduler.step()

        for _ in range(n_iter_critic):
            critic_optimizer.zero_grad()
            critic_loss = critic(h_x.detach()).mean() - critic(y).mean()
            critic_loss.backward()
            critic_optimizer.step()

        if scheduler is not None:
            critic_scheduler.step()

        if plotter:
            figure = plotter.plot_step(x, y, h_x, labels, critic=critic)
            _log(logger, "Transport", figure, advance=False, close=True)

        with torch.no_grad():
            critic_h_x = critic(h_x).mean()
            critic_y = critic(y).mean()
            loss = critic_y + cost_val - critic_h_x

            x = source.sample((n_samples,))
            x_prime = source.sample((n_samples,))
            gw = cost.get_functional(x, x_prime, mover)


        _log(logger, "GW", gw.item(), advance=False)
        _log(logger, "critic(y)", critic_y.item(), advance=False)
        _log(logger, "critic(h_x)", critic_h_x.item(), advance=False)
        _log(logger, "cost", cost_val.item(), advance=False)
        _log(logger, "loss", loss.item())

        progress_widget.set_postfix({"loss": loss.item()})

    if plotter:
        plotter.close_widget()
        figure = plotter.plot_end(source, target, mover, critic=critic)
        _log(logger, "Result", figure, advance=False, close=True)


def run_experiment(source: CompositeDistribution, target: BasicDistribution,
                   mover: nn.Module, critic: nn.Module, cost: Cost, *,
                   logger=None, **kwargs):
    if logger:
        logger.start()
        logger.log_hparams(kwargs)
    try:
        train(source, target, mover, critic, cost, logger=logger, **kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        if logger:
            logger.finish()
