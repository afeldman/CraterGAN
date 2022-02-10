from typing import OrderedDict

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
                lr:float=1e-4,
                latent_dim:int=100,
                b1:float = 0.5,
                b2:float = 0.999,
                **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # networks
        self.data_shape = (channel, width, height)

        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=self.data_shape)
        self.discriminator = Discriminator(img_shape=self.data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.sample_input_img = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        lr=self.hparams.lr

        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        return [opt_g, opt_d], []

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)
            
    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)

        grid = torchvision.utils.make_grid(sample_imgs)

        # tensorboard
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)        
        if(self.current_epoch==0):
            self.logger.experiment.add_graph(
                CraterGAN(self.data_shape[0],self.data_shape[2],self.data_shape[1]), 
                z.detach().to("cpu"))
        self.custom_histogram_adder()

    def generator_loss(self, x):
        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_dim, device=self.device)
        y = torch.ones(x.size(0), 1, device=self.device)

        # generate images
        generated_imgs = self(z)

        D_output = self.discriminator(generated_imgs)

        # ground truth result (ie: all real)
        g_loss = F.binary_cross_entropy(D_output, y)

        return g_loss

    def discriminator_loss(self, x):
        # train discriminator on real
        b = x.size(0)
        x_real = x.view(b, -1)
        y_real = torch.ones(b, 1, device=self.device)

        # calculate real score
        D_output = self.discriminator(x_real)
        D_real_loss = F.binary_cross_entropy(D_output, y_real)

        # train discriminator on fake
        z = torch.randn(b, self.hparams.latent_dim, device=self.device)
        x_fake = self(z)
        y_fake = torch.zeros(b, 1, device=self.device)

        # calculate fake score
        D_output = self.discriminator(x_fake)
        D_fake_loss = F.binary_cross_entropy(D_output, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss

        return D_loss
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch[:-1]

        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_step(x)

        return result

    def generator_step(self, x):
        g_loss = self.generator_loss(x)

        # log to prog bar on each step AND for the full epoch
        # use the generator loss for checkpointing
        self.log("g_loss", g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, x):
        # Measure discriminator's ability to classify real from generated samples
        d_loss = self.discriminator_loss(x)

        # log to prog bar on each step AND for the full epoch
        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True)
        return d_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch[:-1]

        b = x.size(0)
        x_real = x.view(b, -1)
        y_real = torch.ones(b, 1, device=self.device)
        loss = self.discriminator(x_real)
        val_loss = F.binary_cross_entropy(loss, y_real)

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss
