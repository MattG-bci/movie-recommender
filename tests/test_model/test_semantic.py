import pytest

from model.semantic import (
    blend_scores,
    build_movie_profile,
    cosine_similarity,
    normalize_similarity_scores,
    score_semantic_query,
)
from schemas.movie import Movie


def _movie() -> Movie:
    return Movie(
        id=1,
        title="The Example",
        release_year=2024,
        genres=["Drama"],
        director="A Director",
        country="UK",
        actors=["An Actor"],
        description="A known description",
    )


def test_build_movie_profile_uses_only_movie_metadata():
    profile = build_movie_profile(_movie())

    assert profile == (
        "The Example. 2024. A known description. Genres: Drama. "
        "Director: A Director. Cast: An Actor. Country: UK"
    )


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_normalize_similarity_scores_preserves_ties_and_all_ties():
    scores = normalize_similarity_scores({1: 0.2, 2: 0.2, 3: 0.9})

    assert scores[1] == scores[2]
    assert scores[3] == pytest.approx(10.0)
    assert normalize_similarity_scores({1: 0.5, 2: 0.5}) == {1: 5.5, 2: 5.5}


def test_score_semantic_query_normalizes_similarity_to_rating_scale():
    scores = score_semantic_query(
        [1.0, 0.0], {1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.5, 0.5]}
    )

    assert scores == {2: 1.0, 3: 5.5, 1: 10.0}


def test_blend_scores_renormalizes_available_modalities():
    scores = blend_scores({1: 10.0}, text_scores={1: 1.0})

    assert scores == {1: pytest.approx((10.0 * 0.55 + 1.0 * 0.25) / 0.8)}
