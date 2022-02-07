from torch.utils.tensorboard import SummaryWriter
from matplotlib.figure import Figure
from torch import Tensor
import wandb



class Logger:
    def log(self, tag, data, **kwargs):
        raise NotImplementedError


class TensorBoardLogger(Logger):
    def __init__(self, **kwargs):
        self.logger_params = kwargs

    def start(self):
        self.logger = SummaryWriter(**self.logger_params)

    def log(self, tag, data, step=None, **kwargs):
        if isinstance(data, Figure):
            self.logger.add_figure(tag, data, step, **kwargs)
        elif isinstance(data, Tensor):
            self.logger.add_image(tag, data, step, **kwargs)
        elif isinstance(data, float) or isinstance(data, str):
            self.logger.add_scalar(tag, data, step, **kwargs)

    def finish(self):
        self.logger.close()


class WandbLogger(Logger):
    def __init__(self, project, entity):
        self.project = project
        self.entity = entity

    def start(self):
        self.logger = wandb.init(project=self.project,
                                 entity=self.entity)

    def log(self, tag, data, step=None, **kwargs):
        self.logger.log({tag: data}, step)

    def finish(self):
        wandb.finish()
