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

    trainer = Trainer(gpus=gpus, 
                    callbacks=[checkpoint_callback],
                    default_root_dir=checkpoint,
                    logger=logger,
                    auto_lr_find=True,
                    accelerator='ddp', 
                    auto_scale_batch_size='binsearch',
                    auto_select_gpus=True)

    #tuner = Tuner(trainer)
    #model.hparams.batch_size = 0
    #new_batch_size = tuner.scale_batch_size(model, mode="binsearch", init_val=32, max_trials=10, datamodule=model)
    #model.hparams.batch_size = new_batch_size
    #lr_finder = trainer.tuner.lr_find(model,min_lr=1e-08, max_lr=1, mode='linear')

    #model.hparams.lr = lr_finder.suggestion()

    trainer.fit(model, datamodel)

sys.exit(fire.Fire(training))
