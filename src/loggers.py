from torch.utils.tensorboard import SummaryWriter
from matplotlib.figure import Figure
from torch import Tensor
import wandb



class Logger:
    def __init__(self, **kwargs) -> None:
        self.logger_params = kwargs
        self.step = 0

    def advance(self) -> None:
        self.step += 1

    def log(self, tag: str, data, **kwargs):
        raise NotImplementedError


class TensorBoardLogger(Logger):

    def start(self):
        self.logger = SummaryWriter(**self.logger_params)
        self.step = 0

    def log(self, tag, data, advance=True, **kwargs):
        if isinstance(data, Figure):
            self.logger.add_figure(tag, data, self.step, **kwargs)
        elif isinstance(data, Tensor):
            self.logger.add_image(tag, data, self.step, **kwargs)
        elif isinstance(data, float) or isinstance(data, str):
            self.logger.add_scalar(tag, data, self.step, **kwargs)
        if advance: self.advance()

    def finish(self):
        self.logger.close()


class WandbLogger(Logger):

    def start(self, **kwargs):
        self.logger_params.update(**kwargs)
        self.logger = wandb.init(**self.logger_params)
        self.step = 0

    def log_hparams(self, hparams):
        wandb.config.update(hparams)

    def log(self, tag, data, advance=True, **kwargs):
        self.logger.log({tag: data}, self.step)
        if advance: self.advance()

    def finish(self):
        wandb.finish()
