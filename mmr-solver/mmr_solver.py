import typing as tp
from ipywidgets.widgets.widget import Widget

import torch
import torch.nn as nn

from torch import Tensor
from tqdm.auto import trange

import matplotlib.pyplot as plt
from ipywidgets import Output
from IPython.display import clear_output, display


def lp_cost(p: float = 2.) -> tp.Callable[[Tensor, Tensor], Tensor]:
    def func(x1: Tensor, x2: Tensor):
        return torch.norm(x1 - x2, p, dim=1) ** p
    return func


class Plotter:
    def plot_state(self, state: tp.Dict[str, tp.Any]) -> None:
        raise NotImplementedError


class WidgetPlotter(Plotter):
    def init_plotter(self) -> None:
        self.output = Output()
        display(self.output)

    def plot_state(self, state) -> None:
        with self.output:
            clear_output(wait=True)
            plt.figure(figsize=(7, 5))
            plt.scatter(*state["x"].detach().cpu().numpy().T,
                        color="blue", label="Source samples", alpha=.2)
            plt.scatter(*state["y"].detach().cpu().numpy().T,
                        color="green", label="Target samples", alpha=.2)
            plt.scatter(*state["h_x"].detach().cpu().numpy().T,
                        color="red", label="Moved samples", alpha=.2)
            plt.legend(loc="upper left")
            plt.show()


class MMR_Solver:
    _optimizers = {"adam": torch.optim.Adam,
                   "sgd": torch.optim.SGD,
                   "rmsprop": torch.optim.RMSprop}
    def __init__(self,
                 critic: nn.Module,
                 mover: nn.Module,
                 cost: tp.Callable[[Tensor, Tensor], Tensor] = None,
                 n_iter: int = 1000,
                 n_mover_iter: int = 15,
                 n_samples: int = 32,
                 l: float = 1.,
                 device: tp.Union[torch.device, str, None] = None,
                 optimizer: str = "adam",
                 optimizer_params: tp.Dict[str, tp.Any] = {},
                 plotter: tp.Optional[Plotter] = None) -> None:
        plotter = plotter or WidgetPlotter()
        cost = cost or lp_cost()
        self._save_hparams(locals())

        self.critic.to(self.device)
        self.mover.to(self.device)
        self.critic_opt = self._optimizers[optimizer](self.critic.parameters(),
                                                      **optimizer_params)
        self.mover_opt = self._optimizers[optimizer](self.mover.parameters(),
                                                     **optimizer_params)

    def _save_hparams(self, locals: tp.Dict[str, tp.Any]) -> None:
        for argname, argval in locals.items():
            if argname == "self":
                continue
            setattr(self, argname, argval)

    def fit(self, source: torch.distributions.Distribution,
                  target: torch.distributions.Distribution,
                  plot_interval: tp.Optional[int] = 20,
                  verbose: bool = True) -> None:
        if plot_interval:
            self.plotter.init_plotter()
        progress = trange(self.n_iter, disable=not verbose)
        for step in progress:
            x = source.sample((self.n_samples,)).to(self.device)
            y = target.sample((self.n_samples,)).to(self.device)

            for _ in range(self.n_mover_iter):
                h_x = self.mover(x)
                self.mover_opt.zero_grad()
                mover_loss = self.l * self.cost(x, h_x).mean() \
                           - self.critic(h_x).mean()
                mover_loss.backward()
                self.mover_opt.step()

            self.critic_opt.zero_grad()
            critic_loss = self.critic(h_x.detach()).mean() \
                        - self.critic(y).mean()
            critic_loss.backward()
            self.critic_opt.step()

            with torch.no_grad():
                loss = self.critic(y).mean() \
                     + self.l * self.cost(x, h_x).mean() \
                     - self.critic(h_x).mean()
                progress.set_postfix({"loss": loss.item()})

            if plot_interval and step % plot_interval == 0:
                state = dict(step=step, x=x, y=y, h_x=h_x, loss=loss.item())
                self.plotter.plot_state(state)

