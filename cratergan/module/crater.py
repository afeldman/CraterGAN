from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from craterdata.mooncraterdataset import MoonCraterDataset


class CaterDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "./", 
                 num_worker:int=8, 
                 batch_size:int=256):

        super().__init__()

        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.num_worker = num_worker
        self.batch_size = batch_size

        self.save_hyperparameters()

        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 256, 256)


    def prepare_data(self):
        # download data if not available
        MoonCraterDataset(self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None):

        moon_crater = MoonCraterDataset(self.data_dir, transform=self.transform, target_transform=self.transform)

        if stage == "fit" or stage is None:
            crater_split = int(len(moon_crater) * 0.75)
            self.moon_crater_train, self.moon_crater_val = random_split(moon_crater, [crater_split, len(moon_crater) - crater_split])

        # Assign test dataset for use in dataloader(s). the test data set is same as train
        if stage == "test" or stage is None:
            crater_split = int(len(self.moon_crater) * 0.25)
            self.moon_crater_test, _ = random_split(moon_crater, [crater_split, len(moon_crater) - crater_split])

    def train_dataloader(self):
        return DataLoader(self.moon_crater_train, 
                         batch_size=self.hparams.batch_size, 
                         num_workers=self.num_worker)

    def val_dataloader(self):
        return DataLoader(self.moon_crater_val, 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.num_worker)

    def test_dataloader(self):
        return DataLoader(self.moon_crater_test, 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.num_worker)
