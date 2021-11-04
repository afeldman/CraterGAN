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
             batchsize:int = 32,
             checkpoint:str="./checkpoint"):

    checkpoint = ModelCheckpoint(dirpath=f"{checkpoint}/log/",
                                 verbose=True,
                                 monitor="val_acc",
                                 mode="max")

    logger = TensorBoardLogger(f'{checkpoint}/logs/')

    datamodel = CaterDataModule(data_dir=datasource, 
                                num_worker=workers, 
                                batch_size=batchsize)

    image_size = datamodel.size()

    model = CraterGAN(batch_size=batchsize, 
                    channel=image_size[0],
                    height=image_size[1],
                    width=image_size[2])

    train = Trainer(gpus=gpus, 
                    checkpoint_callback=checkpoint,
                    progress_bar_refresh_rate=20, 
                    default_root_dir=checkpoint,
                    logger=logger)

    train.fit(model, datamodel)

sys.exit(fire.Fire(training))
