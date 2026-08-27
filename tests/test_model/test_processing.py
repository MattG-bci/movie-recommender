from datetime import datetime

import pytest
import torch

from model.processing import (
    build_movie_candidates,
    get_model_id_to_recommender_id_mapping,
    map_db_ids_to_recommender_ids,
)
from schemas.movie import Movie
from schemas.users import User


def _make_movies():
    return [
        Movie(
            id=10,
            title="Movie A",
            release_year=2024,
            genres=["Action"],
            director="Director A",
            country="US",
            actors=["Actor A"],
        ),
        Movie(
            id=20,
            title="Movie B",
            release_year=2024,
            genres=["Drama"],
            director="Director B",
            country="UK",
            actors=["Actor B"],
        ),
    ]


def _make_users():
    return [
        User(
            id=100,
            username="alice",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        ),
        User(
            id=200,
            username="bob",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        ),
    ]


def test_get_model_id_to_recommender_id_mapping():
    movies = _make_movies()
    result = get_model_id_to_recommender_id_mapping(movies, "id")

    assert set(result.keys()) == {10, 20}
    assert set(result.values()) == {0, 1}


def test_get_model_id_to_recommender_id_mapping_single_item():
    movies = _make_movies()[:1]
    result = get_model_id_to_recommender_id_mapping(movies, "id")

    assert result == {10: 0}


def test_get_model_id_to_recommender_id_mapping_empty_list():
    result = get_model_id_to_recommender_id_mapping([], "id")
    assert result == {}


def test_prepare_recommendation_mappings():
    movies = _make_movies()
    users = _make_users()

    user_ids, movie_ids = map_db_ids_to_recommender_ids(movies, users)

    assert len(movie_ids) == 2
    assert user_ids == [0, 1]
    assert movie_ids == [0, 1]


def test_build_movie_candidates():
    movies = _make_movies()
    scores = torch.tensor([8.5, 7.0])

    candidates = build_movie_candidates([10, 20], scores, movies)

    assert len(candidates) == 2
    assert candidates[0].movie.id == 10
    assert candidates[0].cf_score == pytest.approx(8.5)
    assert candidates[1].movie.id == 20
    assert candidates[1].cf_score == pytest.approx(7.0)


def test_build_movie_candidates_skips_missing_movie_ids():
    movies = _make_movies()
    scores = torch.tensor([8.5, 7.0])

    candidates = build_movie_candidates([10, 999], scores, movies)

    assert len(candidates) == 1
    assert candidates[0].movie.id == 10


def test_build_movie_candidates_empty_recommendations():
    movies = _make_movies()
    scores = torch.tensor([])

    candidates = build_movie_candidates([], scores, movies)

    assert candidates == []
