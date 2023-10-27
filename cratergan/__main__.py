#!/usr/bin/env python3

import os
import sys
import torch

import fire

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from cratergan.module.crater import CaterDataModule
from cratergan.gan import CraterGAN
from cratergan.__version__ import __version__

def training(datasource:str=".",
             #c_gpus:int=torch.cuda.device_count(), 
             workers:int=os.cpu_count()//2,
             checkpoint:str="./checkpoint",
             batch_size:int = 256,
             checkpoint_file:str = "", 
             strategy=None):

    checkpoint_callback = ModelCheckpoint(dirpath=f"{checkpoint}/log/",
                                 monitor="val_loss",
                                 verbose=True,
                                 save_top_k=3,
                                 mode="min",
                                 save_last=True,
                                 filename='CraterGAN-{epoch:04d}-{val_loss:.5f}')

    logger = TensorBoardLogger(f'{checkpoint}/tb/', 
                                version='cratergan_'+".".join([str(v) for v in __version__]))

    datamodel = CaterDataModule(data_dir=datasource, 
                                num_worker=workers,
                                batch_size=batch_size)

    image_size = datamodel.dims

    if not checkpoint_file or not bool(checkpoint_file.strip()):
        model = CraterGAN(channel=image_size[0],
                    height=image_size[1],
                    width=image_size[2])
    else:
        model = CraterGAN.load_from_checkpoint(f"{checkpoint}/log/{checkpoint_file}")

    if strategy is not None:
        trainer = Trainer(callbacks=[checkpoint_callback],
                    default_root_dir=checkpoint,
                    logger=logger,
                    strategy=strategy,max_epochs=-1)
    else:
        trainer = Trainer(callbacks=[checkpoint_callback],
                    default_root_dir=checkpoint,
                    logger=logger,max_epochs=-1)

    trainer.fit(model=model, datamodule=datamodel)

sys.exit(fire.Fire(training))
