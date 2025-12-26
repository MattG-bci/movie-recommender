from datetime import datetime

from model.train import preprocess_movie_ratings
from schemas.movie import MovieRating, Movie
from schemas.users import User


def test_preprocess_movie_ratings():
    mock_ratings = [
        MovieRating(
            id=7,
            user_id=1,
            movie_id=1,
            rating=3.0,
        ),
        MovieRating(id=8, user_id=3, movie_id=2, rating=5.0),
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
        MovieRating(
            id=7,
            user_id=0,
            movie_id=0,
            rating=3.0,
        ),
        MovieRating(id=8, user_id=2, movie_id=1, rating=5.0),
    ]
    assert result == expected
