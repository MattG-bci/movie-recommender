import torch
from torch import nn


class Recommender(nn.Module):
    def __init__(self, n_movies, n_users):
        super().__init__()
        self.movie_embedding = nn.Embedding(n_movies, 64)
        self.user_embedding = nn.Embedding(n_users, 64)
        self.head = nn.Linear(
            self.user_embedding.embedding_dim + self.movie_embedding.embedding_dim, 1
        )
        self.loss = nn.MSELoss()

    def forward(self, users, movies):
        out_movies = self.movie_embedding(movies)
        out_users = self.user_embedding(users)
        out = torch.concat([out_movies, out_users])
        out = self.head(out)
        return out
