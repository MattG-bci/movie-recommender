import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class Recommender(nn.Module):
    def __init__(self, n_movies: int, n_users: int) -> None:
        super().__init__()
        self.movie_embedding = nn.Embedding(n_movies, 64)
        self.user_embedding = nn.Embedding(n_users, 64)
        self.head = nn.Linear(
            self.user_embedding.embedding_dim + self.movie_embedding.embedding_dim, 1
        )
        self.loss = nn.MSELoss()

        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
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
        return torch.optim.Adam(self.parameters(), lr=0.01)
