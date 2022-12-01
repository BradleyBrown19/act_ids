import pdb
import torch
from pytorch_lightning import LightningDataModule
from hydra.utils import instantiate
from random import sample

class CIFAR(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cifar_train = instantiate(config.cifar_train_args)
        self.cifar_val = instantiate(config.cifar_val_args)

    def setup(self, stage=None):
        self.cifar_train = instantiate(self.config.cifar_train_args)
        self.cifar_val = instantiate(self.config.cifar_val_args)

        if self.config.train_data_pct < 1:
            num_datapoints = self.cifar_train.data.shape[0]
            indices = sample([i for i in range(num_datapoints)], int(num_datapoints*self.config.train_data_pct))

            self.cifar_train.data = self.cifar_train.data[indices]
            self.cifar_train.targets = [self.cifar_train.targets[i] for i in indices]

    def train_dataloader(self):
        return instantiate(self.config.train_dataloader, dataset=self.cifar_train)

    def val_dataloader(self):
        return instantiate(self.config.val_dataloader, dataset=self.cifar_val)
