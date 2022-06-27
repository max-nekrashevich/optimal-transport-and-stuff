import typing as tp

import torch
import torch.nn as nn
import torch.optim as o

from tqdm.auto import tqdm

from .costs import Cost
from .distributions import BasicDistribution, CompositeDistribution
from .loggers import Logger
from .plotters import Plotter
from .utils import filter_dict, _init_opt_or_sch


class Experiment:
    def __init__(self, mover: nn.Module, critic: nn.Module, cost: Cost,
                 source: CompositeDistribution, target: BasicDistribution, *,
                 num_samples: int,
                 num_steps_train: int = 50,
                 num_steps_eval: int = 50,
                 source_eval: tp.Optional[CompositeDistribution] = None,
                 target_eval: tp.Optional[BasicDistribution] = None,
                 num_steps_mover: int = 10,
                 num_steps_critic: int = 1,
                 num_steps_cost: int = 0,
                 nested_train_step: bool = False,
                 alpha: float = .05) -> None:
        self.mover = mover
        self.critic = critic
        self.cost = cost

        self.num_samples = num_samples
        self.num_steps_train = num_steps_train
        self.num_steps_eval = num_steps_eval
        self.num_steps_mover = num_steps_mover
        self.num_steps_critic = num_steps_critic
        self.num_steps_cost = num_steps_cost
        self.nested_train_step = nested_train_step
        self.alpha = alpha

        self.source = source
        self.target = target
        self.source_eval = source_eval or source
        self.target_eval = target_eval or target
        self.x_eval, self.labels_eval = self.source_eval.sample(
            (self.num_samples,), return_labels=True)
        self.y_eval = self.target_eval.sample((self.num_samples,))

        self._init_optimizers = False
        self._init_schedulers = False

        self.train_steps_total = 0
        self.eval_steps_total = 0

    def init_optimizers(self, type: type = o.Adam,
                        params: tp.Dict = dict(lr=5e-5), *,
                        type_mover: tp.Optional[type] = None,
                        type_critic: tp.Optional[type] = None,
                        type_cost: tp.Optional[type] = None,
                        params_mover: tp.Dict = dict(),
                        params_critic: tp.Dict = dict(),
                        params_cost: tp.Dict = dict()) -> "Experiment":
        self.mover_optimizer = _init_opt_or_sch(
            type_mover, params_mover, self.mover.parameters(), type, params)

        self.critic_optimizer = _init_opt_or_sch(
            type_critic, params_critic, self.critic.parameters(), type, params)

        if list(self.cost.parameters()):
            self.cost_optimizer = _init_opt_or_sch(
                type_cost, params_cost, self.cost.parameters(), type, params)
        else:
            self.cost_optimizer = None

        self._init_optimizers = True
        return self

    def init_schedulers(self, type: type, params: tp.Dict, *,
                        type_mover: tp.Optional[type] = None,
                        type_critic: tp.Optional[type] = None,
                        type_cost: tp.Optional[type] = None,
                        params_mover: tp.Dict = dict(),
                        params_critic: tp.Dict = dict(),
                        params_cost: tp.Dict = dict()) -> "Experiment":
        self.mover_scheduler = _init_opt_or_sch(
            type_mover, params_mover, self.mover_optimizer, type, params)

        self.critic_scheduler = _init_opt_or_sch(
            type_critic, params_critic, self.critic.parameters(), type, params)

        if self.cost_optimizer is not None:
            self.cost_scheduler = _init_opt_or_sch(
                type_cost, params_cost, self.cost.parameters(), type, params)

        self._init_schedulers = True
        return self

    @torch.no_grad()
    def compute_losses(self, x: torch.Tensor, y: torch.Tensor) -> tp.Dict:
        h_x = self.mover(x)

        cost = self.alpha * self.cost(x, h_x).mean().item()
        critic_h_x = self.critic(h_x).mean().item()
        critic_y = self.critic(y).mean().item()
        loss = critic_y + cost - critic_h_x
        return {"train/cost": cost,
                "train/critic(h_x)": critic_h_x,
                "train/critic(y)": critic_y,
                "train/loss": loss}

    @torch.no_grad()
    def compute_metrics(self, x: torch.Tensor, y: torch.Tensor) -> tp.Dict:
        x_prime = self.source.sample((self.num_samples,))
        gw = self.cost.get_functional(x, x_prime, self.mover).item()
        # TODO
        return {"eval/GW": gw}

    def _train_step_sequential(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for _ in range(self.num_steps_cost):
            self.cost_optimizer.zero_grad()  # type: ignore
            h_x = self.mover(x)
            cost = self.alpha * self.cost(x, h_x).mean()
            cost.backward()
            self.cost_optimizer.step()  # type: ignore

        for _ in range(self.num_steps_mover):
            self.mover_optimizer.zero_grad()
            h_x = self.mover(x)
            cost = self.alpha * self.cost(x, h_x).mean()
            mover_loss = cost - self.critic(h_x).mean()
            mover_loss.backward()
            self.mover_optimizer.step()

        for _ in range(self.num_steps_critic):
            self.critic_optimizer.zero_grad()
            critic_loss = self.critic(h_x.detach()).mean() - \
                self.critic(y).mean()
            critic_loss.backward()
            self.critic_optimizer.step()

    def _train_step_nested(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for _ in range(self.num_steps_critic):

            for _ in range(self.num_steps_mover):

                for _ in range(self.num_steps_cost):
                    self.cost_optimizer.zero_grad()  # type: ignore
                    h_x = self.mover(x)
                    cost = self.alpha * self.cost(x, h_x).mean()
                    cost.backward()
                    self.cost_optimizer.step()  # type: ignore

                self.mover_optimizer.zero_grad()
                h_x = self.mover(x)
                cost = self.alpha * self.cost(x, h_x).mean()
                mover_loss = cost - self.critic(h_x).mean()
                mover_loss.backward()
                self.mover_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss = self.critic(h_x.detach()).mean() - \
                self.critic(y).mean()
            critic_loss.backward()
            self.critic_optimizer.step()

    def _train_epoch(self, progress, logger, plotter) -> None:
        self.mover.train()
        self.critic.train()
        self.cost.train()

        for step in range(self.num_steps_train):
            x, labels = self.source.sample((self.num_samples,),
                                           return_labels=True)
            y = self.target.sample((self.num_samples,))
            if self.nested_train_step:
                self._train_step_nested(x, y)
            else:
                self._train_step_sequential(x, y)
            losses = self.compute_losses(x, y)

            progress.update()
            progress.set_postfix(filter_dict(losses, ["loss"]))
            logger.log("train/step", self.train_steps_total + step,
                       commit=False)
            logger.log_dict(losses)

        self.train_steps_total += self.num_steps_train

        if plotter is not None:
            with torch.no_grad():
                x, labels = self.source.sample((self.num_samples,),
                                               return_labels=True)
                y = self.target.sample((self.num_samples,))
                h_x = self.mover(x)
            figure = plotter.plot_step(x, y, h_x, labels,
                                       critic=self.critic)
            logger.log("train/transport", figure, close=True,
                       step=self.train_steps_total)

        if self._init_schedulers:
            self.mover_scheduler.step()
            self.critic_scheduler.step()
            if self.cost_optimizer is not None:
                self.cost_scheduler.step()

    @torch.no_grad()
    def _eval_epoch(self, progress, logger, plotter) -> None:
        self.mover.eval()
        self.critic.eval()
        self.cost.eval()

        for step in range(self.num_steps_eval):
            x = self.source_eval.sample((self.num_samples,))
            y = self.target_eval.sample((self.num_samples,))
            metrics = self.compute_metrics(x, y)

            progress.update()
            progress.set_postfix(filter_dict(metrics, ["GW"]))
            logger.log("eval/step", self.eval_steps_total + step,
                       commit=False)
            logger.log_dict(metrics)

        self.eval_steps_total += self.num_steps_eval

        if plotter is not None:
            h_x = self.mover(self.x_eval)
            figure = plotter.plot_step(self.x_eval, self.y_eval, h_x,
                                       self.labels_eval, critic=self.critic)
            logger.log("eval/transport", figure, close=True,
                       step=self.eval_steps_total)

    def run(self, num_epochs: int, *,
            logger: Logger = Logger(),
            plotter: tp.Optional[Plotter] = None,
            show_progress: bool = True) -> None:
        assert self._init_optimizers, \
            "Optimizers not initialized. Initialize using `init_optimizers`."
        assert (self.cost_optimizer is None) or (self.num_steps_cost == 0), \
            "Cost is not optimizable, so `num_steps_cost` mus be set to zero."

        epoch_progress = tqdm(range(num_epochs), desc="Epoch",
                              disable=not show_progress)
        train_progress = tqdm(total=self.num_steps_train, desc="Training",
                              disable=not show_progress, leave=False)
        eval_progress = tqdm(total=self.num_steps_eval, desc="Validating",
                             disable=not show_progress, leave=False)

        logger.start()

        try:
            if plotter:
                plotter.init_widget()

            for _ in epoch_progress:
                self._train_epoch(train_progress, logger, plotter)
                self._eval_epoch(eval_progress, logger, plotter)
                train_progress.reset()
                eval_progress.reset()

            if plotter:
                plotter.close_widget()
                figure = plotter.plot_end(self.source, self.target,
                                          self.mover, critic=self.critic)
                logger.log("train/result", figure, close=True)
                figure = plotter.plot_end(self.source_eval, self.target_eval,
                                          self.mover, critic=self.critic)
                logger.log("eval/result", figure, close=True)

        except KeyboardInterrupt:
            pass
        finally:
            train_progress.close()
            eval_progress.close()
            logger.finish()


def run_experiment(source: CompositeDistribution, target: BasicDistribution,
                   mover: nn.Module, critic: nn.Module, cost: Cost, *,
                   num_epochs: int,
                   logger=Logger(), plotter=None, show_progress=True,
                   optimizer_params=dict(),
                   scheduler_params=dict(),
                   **init_kwargs):
    experiment = Experiment(mover, critic, cost, source, target,
                            **init_kwargs)
    experiment.init_optimizers(**optimizer_params)
    if scheduler_params:
        experiment.init_schedulers(**scheduler_params)
    experiment.run(num_epochs,
                   logger=logger,
                   plotter=plotter,
                   show_progress=show_progress)
