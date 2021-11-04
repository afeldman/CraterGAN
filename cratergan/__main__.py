#!/usr/bin/env python3

import os
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint



import fire

from cratergan.module.crater import CaterDataModule
from cratergan.gan import CraterGAN

def training(datasource:str=".",
             gpus:int=torch.cuda.device_count(), 
             workers:int=os.cpu_count()//2,
             checkpoint:str="./checkpoint"):

    checkpoint_callback = ModelCheckpoint(dirpath=f"{checkpoint}/log/",
                                 verbose=True,
                                 monitor="val_acc",
                                 mode="max")

    logger = TensorBoardLogger(f'{checkpoint}/logs/')

    datamodel = CaterDataModule(data_dir=datasource, 
                                num_worker=workers)

    image_size = datamodel.size()

    model = CraterGAN(channel=image_size[0],
                    height=image_size[1],
                    width=image_size[2])

    train = Trainer(gpus=gpus, 
                    callbacks=[checkpoint_callback],
                    default_root_dir=checkpoint,
                    logger=logger,
                    auto_scale_batch_size=True,
                    auto_lr_find=True)

    train.fit(model, datamodel)

sys.exit(fire.Fire(training))
