import torch
from pydantic import BaseModel

from schemas.movie import Movie, MovieRatingWithId
from schemas.recommendation import MovieCandidate
from schemas.users import User


def prepare_recommendation_mappings(
    movies: list[Movie], user_names: list[User], user_name: str
) -> tuple[dict[int, int], int, list[int], int, int]:
    map_movie_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        movies, "id"
    )
    map_user_id_to_recommender_id = get_model_id_to_recommender_id_mapping(
        user_names, "id"
    )
    map_recommender_id_to_movie_id = {
        value: key for key, value in map_movie_id_to_recommender_id.items()
    }

    map_user_name_to_db_id = {user.username: user.id for user in user_names}

    movie_ids = list({map_movie_id_to_recommender_id[movie.id] for movie in movies})
    n_movies = len(movie_ids)
    n_users = len({user.id for user in user_names})

    database_user_id = map_user_name_to_db_id.get(user_name)
    if database_user_id is None:
        raise KeyError(f"User name {user_name} does not exist in the database")
    recommender_user_id = map_user_id_to_recommender_id[database_user_id]

    return (
        map_recommender_id_to_movie_id,
        recommender_user_id,
        movie_ids,
        n_users,
        n_movies,
    )


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
