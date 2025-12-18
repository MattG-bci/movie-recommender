import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from model.recommender import Recommender


@dataclass
class ConfigTrain:
    model: Recommender
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    device: torch.device
