import torch
from torch import nn, optim
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

        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.movie_embedding.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_ids: list[int], movie_ids: list[int]) -> torch.Tensor:
        user_vecs = self.user_embedding(user_ids)
        movie_vecs = self.movie_embedding(movie_ids)

        dot = (user_vecs * movie_vecs).sum(dim=1, keepdim=True)

        preds = dot + self.user_bias(user_ids) + self.movie_bias(movie_ids)
        return preds.squeeze()


if __name__ == "__main__":
    num_users = 1000
    num_movies = 5000
    model = Recommender(num_users, num_movies)

    user_ids = torch.tensor([1] * 32)
    movie_ids = torch.tensor([1] * 32)
    ratings = torch.rand(32) * 10

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = model.loss

    for epoch in range(5):
        preds = model(user_ids, movie_ids)
        loss = criterion(preds, ratings)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info(f"Training loss: {loss.item():.4f}")
