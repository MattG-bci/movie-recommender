from src.model.recommender import prepare_model_config
from src.schemas.movie import MovieRating


def test_prepare_model_config():
    ratings = [
        MovieRating(id=0, user_id=0, movie_id=0, rating=4.0),
        MovieRating(id=1, user_id=0, movie_id=1, rating=3.5),
        MovieRating(id=2, user_id=1, movie_id=0, rating=5.0),
        MovieRating(id=3, user_id=1, movie_id=2, rating=2.0),
        MovieRating(id=4, user_id=2, movie_id=1, rating=4.5),
    ]

    model_config = prepare_model_config(ratings)
    assert model_config.n_users == 3
    assert model_config.n_movies == 3
