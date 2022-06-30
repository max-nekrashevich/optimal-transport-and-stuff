import typing as tp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as o
from tqdm.auto import tqdm

from .costs import Cost
from .distributions import BasicDistribution, CompositeDistribution
from .loggers import Logger
from .plotters import Plotter
from .utils import (_init_opt_or_sch,
                    calculate_frechet_distance,
                    filter_dict,
                    get_inception_statistics)


class Experiment:
    def __init__(self, mover: nn.Module, critic: nn.Module, cost: Cost,
                 source: CompositeDistribution, target: BasicDistribution, *,
                 source_eval: tp.Optional[CompositeDistribution] = None,
                 target_eval: tp.Optional[BasicDistribution] = None,
                 num_samples: int = 64,
                 num_steps_train: int = 50,
                 num_steps_eval: int = 50,
                 num_steps_mover: int = 10,
                 num_steps_critic: int = 1,
                 num_steps_cost: int = 0,
                 nested_train_step: bool = False,
                 alpha: float = .05,
                 use_fid: bool = True,
                 fid_mu: tp.Optional[np.ndarray] = None,
                 fid_sigma: tp.Optional[np.ndarray] = None,
                 optimizer: type = o.Adam,
                 optimizer_params: tp.Dict = dict(lr=5e-5),
                 optimizer_mover: tp.Optional[type] = None,
                 optimizer_critic: tp.Optional[type] = None,
                 optimizer_cost: tp.Optional[type] = None,
                 optimizer_params_mover: tp.Dict = dict(),
                 optimizer_params_critic: tp.Dict = dict(),
                 optimizer_params_cost: tp.Dict = dict(),
                 ) -> None:
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
        self.compute_fid = use_fid

        self.source = source
        self.target = target
        self.source_eval = source_eval or source
        self.target_eval = target_eval or target

        self.train_steps_total = 0
        self.epochs_total = 0
        self.x_eval, self.labels_eval = self.source_eval.sample(
            (self.num_samples,), return_labels=True)
        self.y_eval = self.target_eval.sample((self.num_samples,))

        if use_fid:
            mu_eval = None
            sigma_eval = None
            if fid_mu is None or fid_sigma is None:
                mu_eval, sigma_eval = get_inception_statistics(
                    self.target_eval.features,  # type: ignore
                    self.num_samples, verbose=True)
            self.mu_eval = mu_eval if fid_mu is None else fid_mu
            self.sigma_eval = sigma_eval if fid_sigma is None else fid_sigma

        self.mover_optimizer = _init_opt_or_sch(
            optimizer_mover, optimizer_params_mover,
            self.mover.parameters(), optimizer, optimizer_params)

        self.critic_optimizer = _init_opt_or_sch(
            optimizer_critic, optimizer_params_critic,
            self.critic.parameters(), optimizer, optimizer_params)

        if list(self.cost.parameters()):
            self.cost_optimizer = _init_opt_or_sch(
                optimizer_cost, optimizer_params_cost,
                self.cost.parameters(), optimizer, optimizer_params)
        else:
            self.cost_optimizer = None

        self._init_schedulers = False

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

        h_x = self.mover(x)

        for _ in range(self.num_steps_critic):
            self.critic_optimizer.zero_grad()
            critic_loss = self.critic(h_x.detach()).mean() - \
                self.critic(y).mean()
            critic_loss.backward()
            self.critic_optimizer.step()

    def _train_step_nested(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for _ in range(self.num_steps_critic):
            h_x = self.mover(x)

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

        GWs: tp.List[float] = []

        h_xs: tp.List[torch.Tensor] = []

        for _ in range(self.num_steps_eval):
            x = self.source_eval.sample((self.num_samples,))
            # y = self.target_eval.sample((self.num_samples,))
            h_x = self.mover(x)
            h_xs.append(h_x)

            x_prime = self.source.sample((self.num_samples,))
            h_x_prime = self.mover(x_prime)
            GWs.append(self.cost.get_functional(x, x_prime,
                                                h_x, h_x_prime).item())

            progress.update()
            progress.set_postfix({"GW": GWs[-1]})

        metrics = {"eval/GW": np.mean(GWs)}
        if self.compute_fid:
            mu, sigma = get_inception_statistics(
                torch.cat(h_xs), self.num_samples)
            metrics["eval/FID"] = calculate_frechet_distance(
                mu, sigma, self.mu_eval, self.sigma_eval)

        logger.log_dict(metrics, step=self.epochs_total)

        if plotter is not None:
            h_x = self.mover(self.x_eval)
            figure = plotter.plot_step(self.x_eval, self.y_eval, h_x,
                                       self.labels_eval, critic=self.critic)
            logger.log("eval/transport", figure, close=True,
                       step=self.epochs_total)

    def run(self, num_epochs: int, *,
            logger: Logger = Logger(),
            plotter: tp.Optional[Plotter] = None,
            show_progress: bool = True) -> None:
        assert (self.cost_optimizer is not None) or \
            (self.num_steps_cost == 0), \
            "Cost is not optimizable, so `num_steps_cost` must be set to zero."

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
                logger.log("epoch", self.epochs_total, commit=False)
                self._train_epoch(train_progress, logger, plotter)
                self._eval_epoch(eval_progress, logger, plotter)
                train_progress.reset()
                eval_progress.reset()
                self.epochs_total += 1

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
                   scheduler_params=dict(),
                   **config):
    experiment = Experiment(mover, critic, cost, source, target,
                            **config)
    if scheduler_params:
        experiment.init_schedulers(**scheduler_params)
    experiment.run(num_epochs,
                   logger=logger,
                   plotter=plotter,
                   show_progress=show_progress)
