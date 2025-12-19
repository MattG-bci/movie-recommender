import torch
from torch import nn
import logging

from schemas.modelling import ModelConfig
from schemas.movie import MovieRating

logger = logging.getLogger(__name__)


class Recommender(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.movie_embedding = nn.Embedding(
            self.config.n_movies, self.config.embedding_dim
        )
        self.user_embedding = nn.Embedding(
            self.config.n_users, self.config.embedding_dim
        )
        self.head = nn.Linear(
            self.user_embedding.embedding_dim + self.movie_embedding.embedding_dim, 1
        )
        self.loss = self.config.loss

        self.user_bias = nn.Embedding(self.config.n_users, 1)
        self.movie_bias = nn.Embedding(self.config.n_movies, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.movie_embedding.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_vecs = self.user_embedding(user_ids)
        movie_vecs = self.movie_embedding(movie_ids)

        dot = (user_vecs * movie_vecs).sum(dim=1, keepdim=True)

        preds = dot + self.user_bias(user_ids) + self.movie_bias(movie_ids)
        return preds.squeeze()

    @property
    def optimiser(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)


def prepare_model_config(ratings: list[MovieRating]) -> ModelConfig:
    n_users = len({rating.user_id for rating in ratings})
    n_movies = len({rating.movie_id for rating in ratings})
    model_config = ModelConfig(n_users=n_users, n_movies=n_movies)
    return model_config
