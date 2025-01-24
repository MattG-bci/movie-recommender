import os
import torch
import pytorch_lightning as pl
from utils.config import config
from utils.classes import Singleton
from src.ingestion.infer_db_pd import load_db
from sklearn.model_selection import train_test_split


class MovieDataloader(pl.LightningDataModule, metaclass=Singleton):
    def __init__(self):
        self.data_dir = os.path.join(config["db_dir"], config["db_name"])

    def prepare_data(self):
        return load_db()

    def setup(self, stage: str = None):
        dataset = self.prepare_data()
        self.train_data, self.val_data = train_test_split(
            dataset, train_size=0.7, random_state=42, shuffle=True
        )
        self.val_data, self.test_data = train_test_split(
            self.val_data, train_size=0.5, random_state=42, shuffle=False
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=1)
