from datetime import datetime

from model.train import preprocess_movie_ratings, prepare_train_config_for_cfrecommender
from schemas.movie import MovieRatingWithId, Movie
from schemas.modelling import ModelTrainConfig
from schemas.users import User


def test_preprocess_movie_ratings():
    mock_ratings = [
        MovieRatingWithId(
            id=7,
            user_id=1,
            movie_id=1,
            rating=3.0,
        ),
        MovieRatingWithId(id=8, user_id=3, movie_id=2, rating=5.0),
    ]
    mock_movies = [
        Movie(
            id=1,
            title="test",
            release_year=2024,
            genres=["test"],
            director="test",
            country="test",
            actors=["test"],
        ),
        Movie(
            id=2,
            title="test",
            release_year=2024,
            genres=["test"],
            director="test",
            country="test",
            actors=["test"],
        ),
        Movie(
            id=3,
            title="test",
            release_year=2024,
            genres=["test"],
            director="test",
            country="test",
            actors=["test"],
        ),
    ]
    mock_users = [
        User(
            id=1,
            username="test1",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        ),
        User(
            id=2,
            username="test2",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        ),
        User(
            id=3,
            username="test3",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        ),
    ]
    result = preprocess_movie_ratings(mock_ratings, mock_movies, mock_users)

    expected = [
        MovieRatingWithId(
            id=7,
            user_id=0,
            movie_id=0,
            rating=3.0,
        ),
        MovieRatingWithId(id=8, user_id=2, movie_id=1, rating=5.0),
    ]
    assert result == expected


def test_prepare_train_config_for_cfrecommender():
    mock_movies = [
        Movie(
            id=1,
            title="Movie A",
            release_year=2024,
            genres=["Action"],
            director="Director A",
            country="US",
            actors=["Actor A"],
        ),
        Movie(
            id=2,
            title="Movie B",
            release_year=2024,
            genres=["Drama"],
            director="Director B",
            country="UK",
            actors=["Actor B"],
        ),
    ]
    mock_users = [
        User(
            id=1,
            username="user1",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        ),
        User(
            id=2,
            username="user2",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        ),
    ]
    mock_ratings = [
        MovieRatingWithId(id=1, user_id=1, movie_id=1, rating=8.0),
        MovieRatingWithId(id=2, user_id=1, movie_id=2, rating=6.0),
        MovieRatingWithId(id=3, user_id=2, movie_id=1, rating=7.0),
        MovieRatingWithId(id=4, user_id=2, movie_id=2, rating=9.0),
    ]

    result = prepare_train_config_for_cfrecommender(
        mock_users, mock_movies, mock_ratings
    )

    assert isinstance(result, ModelTrainConfig)
    assert len(result.train_dataloader.dataset) + len(
        result.val_dataloader.dataset
    ) == len(mock_ratings)
