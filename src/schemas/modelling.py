import torch
from torch import nn
from dataclasses import dataclass

from torch.utils.data import DataLoader

from model.dataloader import MoviesDataset


PATH_TO_MODEL_WEIGHTS = "models/recommender_model.pth"


@dataclass
class ModelTrainHyperparameters:
    epochs: int = 100
    batch_size: int = 64


@dataclass
class ModelTrainConfig:
    model: nn.Module
    train_dataloader: DataLoader[MoviesDataset]
    val_dataloader: DataLoader[MoviesDataset]
    device: torch.device
    hyperparams: ModelTrainHyperparameters


@dataclass
class ModelConfig:
    n_users: int
    n_movies: int
    embedding_dim: int = 32
    learning_rate: float = 0.01
    loss: nn.Module = nn.MSELoss()
    weight_decay: float = 0.0
    dropout_rate: float = 0.3
