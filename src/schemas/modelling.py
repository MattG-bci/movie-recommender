import torch
from torch import nn
from dataclasses import dataclass

from model.dataloader import MoviesDataset


@dataclass
class TrainConfig:
    model: nn.Module
    train_dataset: MoviesDataset
    val_dataset: MoviesDataset
    device: torch.device
    epochs: int = 25
    batch_size: int = 64


@dataclass
class ModelConfig:
    n_users: int
    n_movies: int
    embedding_dim: int = 64
    learning_rate: float = 0.01
