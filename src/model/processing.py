import torch
from pydantic import BaseModel

from schemas.movie import Movie, MovieRatingWithId
from schemas.recommendation import MovieCandidate
from schemas.users import User


def map_db_ids_to_recommender_ids(
    movies: list[Movie], users: list[User]
) -> tuple[list[int], list[int]]:
    map_movie_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        movies, "id"
    )
    map_user_id_to_recommender_id = get_model_id_to_recommender_id_mapping(users, "id")
    recommender_movie_ids = list(
        {map_movie_id_to_recommender_id[movie.id] for movie in movies}
    )
    recommender_user_ids = list(
        {map_user_id_to_recommender_id[user.id] for user in users}
    )
    return (
        recommender_user_ids,
        recommender_movie_ids,
    )


def map_recommender_movie_ids_to_db_ids(
    movies: list[Movie], recommended_movie_ids: torch.Tensor
) -> list[int]:
    map_movie_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        movies, "id"
    )
    map_recommender_id_to_movie_id = {
        recommender_id: movie_id
        for movie_id, recommender_id in map_movie_id_to_recommender_id.items()
    }
    return [
        map_recommender_id_to_movie_id.get(int(recommended_movie_id))
        for recommended_movie_id in recommended_movie_ids
    ]


def build_movie_candidates(
    recommended_movie_ids: list[int],
    cf_scores: torch.Tensor,
    movies: list[Movie],
) -> list[MovieCandidate]:
    candidates: list[MovieCandidate] = []
    for recommended_movie_id, score in zip(recommended_movie_ids, cf_scores):
        movie = list(filter(lambda x: x.id == recommended_movie_id, movies))
        if not movie:
            continue
        candidate = MovieCandidate(movie=movie[0], cf_score=float(score.detach()))
        candidates.append(candidate)
    return candidates


def get_model_id_to_recommender_id_mapping(
    models: list[BaseModel], id_field_name: str
) -> dict[int, int]:
    ids = {getattr(model, id_field_name) for model in models}
    ids = sorted(ids)
    return {model_id: idx for idx, model_id in enumerate(ids)}


def preprocess_movie_ratings(
    ratings: list[MovieRatingWithId], movies: list[Movie], users: list[User]
) -> list[MovieRatingWithId]:
    map_movie_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        movies, "id"
    )
    map_user_id_to_recommender_id = get_model_id_to_recommender_id_mapping(users, "id")
    ratings = [
        rating.model_copy(
            update={
                "user_id": map_user_id_to_recommender_id[rating.user_id],
                "movie_id": map_movie_id_to_recommender_id[rating.movie_id],
            }
        )
        for rating in ratings
    ]
    return ratings
