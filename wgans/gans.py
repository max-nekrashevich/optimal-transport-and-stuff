import torch
import pytorch_lightning as pl

from hydra.utils import instantiate
from omegaconf import OmegaConf


class GAN(pl.LightningModule):
    def __init__(self, data_dim: tuple, code_size: int,
                 critic: OmegaConf,
                 generator: OmegaConf,
                 noise_distribution: OmegaConf,
                 optimizer_config: OmegaConf,
                 critic_step_period: int = 1,
                 generator_step_period: int = 1) -> None:
        super().__init__()

        self.critic = instantiate(critic, data_dim=data_dim)
        self.generator = instantiate(generator,
                                     data_dim=data_dim,
                                     code_size=code_size)
        self.noise_distribution = instantiate(noise_distribution,
                                               dim=code_size)
        self.optimizer_config = optimizer_config
        self.d_step_period = critic_step_period
        self.g_step_period = generator_step_period


class Wasserstein1GAN(GAN):
    def __init__(self, data_dim: tuple, code_size: int,
                 critic: OmegaConf,
                 generator: OmegaConf,
                 noise_distribution: OmegaConf,
                 optimizer_config: OmegaConf,
                 critic_step_period: int = 1,
                 generator_step_period: int = 1,
                 weight_range: float = 0.01
                 ) -> None:

        super().__init__(data_dim, code_size,
                         critic,
                         generator,
                         noise_distribution,
                         optimizer_config,
                         critic_step_period,
                         generator_step_period)
        self.weight_range = weight_range

    def configure_optimizers(self):
        optimizer_d = instantiate(self.optimizer_config,
                                  params=self.critic.parameters())
        optimizer_g = instantiate(self.optimizer_config,
                                  params=self.generator.parameters())
        return [optimizer_d, optimizer_g], []

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)
        if optimizer_idx == 0:
            for parameter in self.critic.parameters():
                parameter.data.clamp_(-self.weight_range,
                                      self.weight_range)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, _ = batch
        batch_size = real_images.size(0)

        noise = self.noise_distribution.sample(batch_size).type_as(real_images)
        fake_images = self.generator(noise)
        self.logger.experiment.add_images("generated_images", fake_images[:6], self.global_step) # type: ignore

        if optimizer_idx == 0 and batch_idx % self.d_step_period == 0:
            critic_loss = self.critic(fake_images.detach()).mean() - \
                                 self.critic(real_images).mean()
            self.log("critic_loss", critic_loss)
            return critic_loss

        if optimizer_idx == 1 and batch_idx % self.g_step_period == 0:
            generator_loss = -self.critic(fake_images).mean()
            self.log("generator_loss", generator_loss)
            return generator_loss


class ThreePlayerGAN(GAN):
    def __init__(self, data_dim: tuple, code_size: int,
                 critic: OmegaConf,
                 generator: OmegaConf,
                 transport: OmegaConf,
                 noise_distribution: OmegaConf,
                 optimizer_config: OmegaConf,
                 critic_step_period: int = 1,
                 generator_step_period: int = 1,
                 transport_step_period: int = 1,
                 l: float = 1.0,
                 p: float = 2.0,
                 reversed: bool = False) -> None:

        super().__init__(data_dim, code_size,
                         critic,
                         generator,
                         noise_distribution,
                         optimizer_config,
                         critic_step_period,
                         generator_step_period)

        self.transport = instantiate(transport,
                                     data_dim=data_dim)
        self.h_step_period = transport_step_period
        self.l = l
        self.p = p
        self.reversed = reversed

    def configure_optimizers(self):
        optimizer_d = instantiate(self.optimizer_config,
                                  params=self.critic.parameters())
        optimizer_g = instantiate(self.optimizer_config,
                                  params=self.generator.parameters())
        optimizer_h = instantiate(self.optimizer_config,
                                  params=self.transport.parameters())
        return [optimizer_d, optimizer_g, optimizer_h], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, _ = batch
        batch_size = real_images.size(0)

        noise = self.noise_distribution.sample(batch_size).type_as(real_images)
        fake_images = self.generator(noise)
        self.logger.experiment.add_images("generated_images", fake_images[:6], self.global_step) # type: ignore


        if self.reversed:
            fake_images, real_images = real_images, fake_images

        tran_images = self.transport(fake_images)

        with torch.no_grad():
            loss = self.critic(real_images).mean() + \
                self.l * torch.norm(fake_images - tran_images, p=self.p) ** self.p - \
                self.critic(tran_images).mean()
            self.log("loss", loss)

        if optimizer_idx == 0 and batch_idx % self.d_step_period == 0:
            critic_loss = self.critic(tran_images.detach()).mean() - \
                                 self.critic(real_images.detach()).mean()
            self.log("critic_loss", critic_loss)
            return critic_loss

        if optimizer_idx == 1 and batch_idx % self.g_step_period == 0:
            if self.reversed:
                generator_loss = self.critic(real_images).mean()
            else:
                generator_loss = self.l * torch.norm(fake_images - tran_images, p=self.p) ** self.p - \
                                 self.critic(tran_images).mean()
            self.log("generator_loss", generator_loss)
            return generator_loss

        if optimizer_idx == 2 and batch_idx % self.h_step_period == 0:
            transport_loss = self.l * torch.norm(fake_images - tran_images, p=self.p) ** self.p - \
                             self.critic(tran_images).mean()
            self.log("transport_loss", transport_loss)
            return transport_loss
