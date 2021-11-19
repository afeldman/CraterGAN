#!/usr/bin/env python3

import os
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner



import fire

from cratergan.module.crater import CaterDataModule
from cratergan.gan import CraterGAN

def training(datasource:str=".",
             gpus:int=torch.cuda.device_count(), 
             workers:int=os.cpu_count()//2,
             checkpoint:str="./checkpoint", 
             strategy=None):

    checkpoint_callback = ModelCheckpoint(dirpath=f"{checkpoint}/log/",
                                 verbose=True,
                                 save_top_k=3,
                                 mode="max",
                                 filename='CraterGAN-{epoch:02d}')

    logger = TensorBoardLogger(f'{checkpoint}/logs/')

    datamodel = CaterDataModule(data_dir=datasource, 
                                num_worker=workers)

    image_size = datamodel.size()

    model = CraterGAN(channel=image_size[0],
                    height=image_size[1],
                    width=image_size[2])

    trainer = Trainer(gpus=gpus, 
                    callbacks=[checkpoint_callback],
                    default_root_dir=checkpoint,
                    logger=logger,
                    auto_lr_find=True,
                    strategy=strategy,
                    auto_scale_batch_size='binsearch',
                    auto_select_gpus=True)

    trainer.fit(model, datamodel)

sys.exit(fire.Fire(training))
