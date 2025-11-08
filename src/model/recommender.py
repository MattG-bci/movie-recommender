import torch
from torch import nn, optim
import logging

from schemas.movie import MovieRating
from sklearn.model_selection import train_test_split

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


def train_movie_recommender(ratings: list[MovieRating], epochs: int = 100) -> None:
    user_ids = torch.tensor([rating.user_id for rating in ratings])
    movie_ids = torch.tensor([rating.movie_id for rating in ratings])
    rating_values = torch.tensor(
        [rating.rating for rating in ratings], dtype=torch.float
    )
    n_users = len(user_ids)
    n_movies = len(movie_ids)

    features = torch.stack([user_ids, movie_ids], dim=1)
    X_train, X_val, y_train, y_val = train_test_split(
        features, rating_values, test_size=0.3, random_state=42
    )

    model = Recommender(n_users, n_movies)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = model.loss

    for epoch in range(epochs):
        train_preds = model(X_train[:, 0], X_train[:, 1])
        loss = criterion(train_preds, y_train)
        logger.info(f"Training loss: {loss.item():.4f}")

        val_preds = model(X_val[:, 0], X_val[:, 1])
        val_loss = criterion(val_preds, y_val)
        logger.info(f"Validation loss: {val_loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
