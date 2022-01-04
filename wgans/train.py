import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate


@hydra.main(config_path="configs")
def main(config: DictConfig):
    model = instantiate(config.model,
                        optimizer_config=config.optimizer,
                        _recursive_=False)
    datamodule = instantiate(config.datamodule)
    trainer = pl.Trainer(**config.trainer)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
