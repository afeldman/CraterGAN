import torch
import torch.nn.functional as F

import torchvision

from pytorch_lightning import LightningModule

from cratergan.module.generator import Generator
from cratergan.module.discriminator import Discriminator

class CraterGAN(LightningModule):
    def __init__(self, 
                channel:int,
                height:int, 
                width:int, 
                latent_dim:int=100,
                lr:float = 2E-4,
                br1:float = 0.5,
                br2:float = 0.999,
                batch_size:int = 16,
                **kwargs):
        super().__init__()

        # hyperparameter
        self.save_hyperparameters()

        # networks
        data_shape = (channel, width, height)

        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.sample_input_img = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass
