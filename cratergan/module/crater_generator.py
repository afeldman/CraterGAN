from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from craterdata.mooncraterdataset import MoonCraterDataset


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", shuffle:bool=False):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        self.shuffle = shuffle

        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 256, 256)

        self.moon_crater = MoonCraterDataset(self.data_dir, transform=self.transform, target_transform=self.transform)

    def prepare_data(self):
        # download data if not available
        MoonCraterDataset(self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            crater_split = np.floor(len(self.moon_crater) * 0.75)
            self.moon_crater_train, self.moon_crater_val = random_split(self.moon_crater, [crater_split, len(self.moon_crater) - crater_split])

        # Assign test dataset for use in dataloader(s). the test data set is same as train
        if stage == "test" or stage is None:
            crater_split = np.floor(len(self.moon_crater) * 0.25)
            self.moon_crater_test, _ = random_split(self.moon_crater, [crater_split, len(self.moon_crater) - crater_split])

    def train_dataloader(self):
        return DataLoader(self.moon_crater_train, batch_size=16)

    def val_dataloader(self):
        return DataLoader(self.moon_crater_val, batch_size=16)

    def test_dataloader(self):
        return DataLoader(self.moon_crater_test, batch_size=16)
