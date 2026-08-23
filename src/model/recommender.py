import dspy
import torch
from torch import nn
import logging
from typing import Any

from schemas.modelling import ModelConfig, PATH_TO_MODEL_WEIGHTS
from schemas.movie import Movie
from schemas.recommendation import RerankMovies
from schemas.users import User

logger = logging.getLogger(__name__)


class MovieReranker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rerank = dspy.ChainOfThought(RerankMovies)

    def forward(
        self,
        request: str,
        exploration: float,
        user_profile: dict[str, Any],
        candidates: list[dict[str, Any]],
    ):
        return self.rerank(
            request=request,
            exploration=exploration,
            user_profile=user_profile,
            candidates=candidates,
        )


class CFRecommender(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.movie_embedding = nn.Embedding(
            self.config.n_movies, self.config.embedding_dim
        )
        self.user_embedding = nn.Embedding(
            self.config.n_users, self.config.embedding_dim
        )
        self.head = nn.Sequential(
            nn.Linear(self.config.embedding_dim * 3, self.config.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.embedding_dim, 1),
        )

        self.loss = self.config.loss

        self.dropout = nn.Dropout(p=self.config.dropout_rate)
        self.user_bias = nn.Embedding(self.config.n_users, 1)
        self.movie_bias = nn.Embedding(self.config.n_movies, 1)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_vecs = self.dropout(self.user_embedding(user_ids))
        movie_vecs = self.dropout(self.movie_embedding(movie_ids))

        interaction = user_vecs * movie_vecs  # element-wise product
        out = torch.concat([user_vecs, movie_vecs, interaction], dim=-1)
        preds = self.head(out)  # update head input dim to embed_dim * 3
        preds = preds + self.user_bias(user_ids) + self.movie_bias(movie_ids)
        return preds.squeeze()

    def predict(self, user_id: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        preds = self.forward(user_id, movie_ids)
        # ratings only range from 1 to 10
        clamped_preds = torch.clamp(min=1.0, max=10.0, input=preds)
        return clamped_preds.squeeze()

    def get_top_k_recommendations(
        self, user_id: torch.Tensor, movie_ids: torch.Tensor, k: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            user_ids = user_id.expand_as(movie_ids)
            preds = self.predict(user_ids, movie_ids)
        top = torch.topk(preds, k)
        top_movie_ids = movie_ids[top.indices.detach()]
        return top_movie_ids, top.values.detach()

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


def load_cf_recommender(n_users: int, n_movies: int) -> CFRecommender:
    model_config = ModelConfig(n_users=n_users, n_movies=n_movies)
    state_dict = torch.load(PATH_TO_MODEL_WEIGHTS, map_location=torch.device("cpu"))
    model = CFRecommender(model_config)
    model.load_state_dict(state_dict)
    return model


def get_cf_recommendations(
    model: CFRecommender,
    recommender_user_id: int,
    movie_ids: list[int],
    map_recommender_id_to_movie_id: dict[int, int],
    n_cf_candidates: int,
) -> tuple[list[int], torch.Tensor]:
    user_id_tensor = torch.tensor([recommender_user_id]).to(torch.device("cpu"))
    movie_id_tensor = torch.tensor(movie_ids).to(torch.device("cpu"))
    recommendations, cf_scores = model.get_top_k_recommendations(
        user_id_tensor, movie_id_tensor, k=n_cf_candidates
    )

    recommended_movie_ids = [
        map_recommender_id_to_movie_id.get(int(recommended_movie_id))
        for recommended_movie_id in recommendations
    ]
    return recommended_movie_ids, cf_scores
