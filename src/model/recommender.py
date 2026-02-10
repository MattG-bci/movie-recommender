import torch
from pydantic import BaseModel
from torch import nn
import logging

from schemas.modelling import ModelConfig
from schemas.movie import Movie
from schemas.users import User

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

        self.dropout = nn.Dropout(p=0.2)
        self.user_bias = nn.Embedding(self.config.n_users, 1)
        self.movie_bias = nn.Embedding(self.config.n_movies, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_vecs = self.dropout(self.user_embedding(user_ids))
        movie_vecs = self.dropout(self.movie_embedding(movie_ids))

        out = torch.concat([user_vecs, movie_vecs], dim=1)
        preds = self.head(out)

        preds = preds + self.user_bias(user_ids) + self.movie_bias(movie_ids)
        return preds.squeeze()

    def predict(self, user_id: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        preds = self.forward(user_id, movie_ids)
        # ratings only range from 1 to 10
        clamped_preds = torch.clamp(min=1.0, max=10.0, input=preds)
        return clamped_preds.squeeze()

    def get_top_k_recommendations(
        self, user_id: torch.Tensor, movie_ids: torch.Tensor, k: int = 5
    ) -> torch.Tensor:
        preds = self.predict(user_id, movie_ids)
        top_recommendations = torch.topk(preds, k).indices.detach()
        return top_recommendations

    @property
    def optimiser(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )


def prepare_model_config(movies: list[Movie], users: list[User]) -> ModelConfig:
    n_users = len({user.id for user in users})
    n_movies = len({movie.id for movie in movies})
    model_config = ModelConfig(n_users=n_users, n_movies=n_movies)
    return model_config


def get_model_id_to_recommender_id_mapping(
    models: list[BaseModel], id_field_name: str
) -> dict[int, int]:
    ids = {getattr(model, id_field_name) for model in models}
    return {model_id: idx for idx, model_id in enumerate(ids)}
