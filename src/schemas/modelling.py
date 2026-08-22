import torch
from torch import nn
from dataclasses import dataclass

from torch.utils.data import DataLoader

from model.dataloader import MoviesDataset


PATH_TO_MODEL_WEIGHTS = "models/recommender_model.pth"


@dataclass
class TrainHyperparameters:
    epochs: int = 10
    batch_size: int = 64


@dataclass
class TrainConfig:
    model: nn.Module
    train_dataloader: DataLoader[MoviesDataset]
    val_dataloader: DataLoader[MoviesDataset]
    device: torch.device
    hyperparams: TrainHyperparameters


@dataclass
class ModelConfig:
    n_users: int
    n_movies: int
    embedding_dim: int = 64
    learning_rate: float = 0.01
    loss: nn.Module = nn.MSELoss()
    weight_decay: float = 0.0001
