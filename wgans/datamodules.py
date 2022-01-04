# type: ignore

import torchvision.transforms as t

from typing import Callable
from torch import Tensor
from hydra.utils import to_absolute_path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CelebA, MNIST


class CustomDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 transform: Callable[[Tensor], Tensor] = t.ToTensor(),
                 num_workers: int = 4,
                 root: str = "data/",
                 download: bool = True):
        super().__init__()
        root = to_absolute_path(root)
        self.save_hyperparameters()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage=None) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)


class CelebADataModule(CustomDataModule):
    def prepare_data(self) -> None:
        CelebA(self.hparams.root, "all", download=self.hparams.download)

    def setup(self, stage=None) -> None:
        self.hparams.transform = t.Compose([t.Resize(64), t.ToTensor()])
        self.train_dataset = CelebA(self.hparams.root, split="train",
                                    transform=self.hparams.transform)
        self.valid_dataset = CelebA(self.hparams.root, split="valid",
                                    transform=self.hparams.transform)
        self.test_dataset = CelebA(self.hparams.root, split="test",
                                   transform=self.hparams.transform)


class MNISTDataModule(CustomDataModule):
    def prepare_data(self) -> None:
        MNIST(self.hparams.root, train=True, download=self.hparams.download)
        MNIST(self.hparams.root, train=False, download=self.hparams.download)

    def setup(self, stage=None) -> None:
        full_dataset = MNIST(self.hparams.root, train=True,
                            transform=self.hparams.transform)
        self.train_dataset, self.valid_dataset = random_split(full_dataset, [55000, 5000])
        self.test_dataset = MNIST(self.hparams.root, train=False,
                                transform=self.hparams.transform)
