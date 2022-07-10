import typing as tp

import wandb


class Logger:
    def __init__(self, **kwargs) -> None:
        self.logger_params = kwargs

    def start(self) -> None:
        pass

    def log(self, tag: str, data: tp.Any, **kwargs):
        pass

    def log_dict(self, dct: tp.Dict[str, tp.Any], **kwargs):
        pass

    def finish(self) -> None:
        pass

    @property
    def name(self) -> tp.Optional[str]:
        return self.logger_params.get("name")

    @name.setter
    def name(self, name) -> None:
        self.logger_params["name"] = name

    @name.deleter
    def name(self) -> None:
        del self.logger_params["name"]


class WandbLogger(Logger):
    def start(self) -> None:
        self.logger = wandb.init(**self.logger_params)
        wandb.define_metric("train/step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("eval/*", step_metric="epoch")

    def log_hparams(self, hparams) -> None:
        wandb.config.update(hparams)

    def log(self, tag, data, **kwargs) -> None:
        self.logger.log({tag: data})  # type: ignore

    def log_dict(self, dct: tp.Dict[str, tp.Any], **kwargs) -> None:
        self.logger.log(dct)  # type: ignore

    def finish(self) -> None:
        wandb.finish()

    @Logger.name.getter
    def name(self) -> tp.Optional[str]:
        if wandb.run is not None:
            return wandb.run.name
        return super().name
