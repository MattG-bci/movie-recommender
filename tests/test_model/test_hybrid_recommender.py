from datetime import datetime
from uuid import UUID

import pytest
import dspy

from model import hybrid_recommender
from schemas.movie import Movie
from schemas.recommendation import MovieEmbedding, RecommendationInput
from schemas.users import User


def _movies() -> list[Movie]:
    return [
        Movie(
            id=1,
            id_uuid=UUID("00000000-0000-0000-0000-000000000001"),
            title="One",
            release_year=2024,
            genres=["Drama"],
            director="A",
            country="UK",
            actors=["A"],
        ),
        Movie(
            id=2,
            id_uuid=UUID("00000000-0000-0000-0000-000000000002"),
            title="Two",
            release_year=2024,
            genres=["Comedy"],
            director="B",
            country="UK",
            actors=["B"],
        ),
    ]


def _users() -> list[User]:
    timestamp = datetime(2024, 1, 1)
    return [User(id=10, username="user", created_at=timestamp, updated_at=timestamp)]


class _EmbeddingClient:
    def embed(self, texts):
        assert texts == ["funny"]
        return [[1.0, 0.0]]


@pytest.mark.asyncio
async def test_recommend_movies_scores_all_movies_without_reranking(monkeypatch):
    movies = _movies()
    stored = [
        MovieEmbedding(
            movie_id=1,
            movie_uuid=movies[0].id_uuid,
            recommendation_profile="one",
            embedding_model="model",
            embedding=[0.0, 1.0],
        ),
        MovieEmbedding(
            movie_id=2,
            movie_uuid=movies[1].id_uuid,
            recommendation_profile="two",
            embedding_model="model",
            embedding=[1.0, 0.0],
        ),
    ]

    async def load_data(_):
        return movies, _users(), stored

    monkeypatch.setattr(
        hybrid_recommender, "load_data_from_db_for_recommendation", load_data
    )
    monkeypatch.setattr(hybrid_recommender, "load_cf_recommender", lambda *_: object())
    monkeypatch.setattr(
        hybrid_recommender,
        "get_cf_recommendations",
        lambda *args: (
            __import__("torch").tensor([0, 1]),
            __import__("torch").tensor([10.0, 1.0]),
        ),
    )
    monkeypatch.setattr(
        hybrid_recommender,
        "EmbeddingSettings",
        lambda: type("S", (), {"MODEL": "model"})(),
    )
    monkeypatch.setattr(hybrid_recommender, "EmbeddingClient", _EmbeddingClient)

    result = await hybrid_recommender.recommend_movies(
        RecommendationInput(username="user", prompt="funny", top_k_recommendations=2)
    )

    assert [item.movie.id for item in result] == [1, 2]
    assert result[0].reason is None
    assert result[0].match_score == pytest.approx((10 * 0.55 + 1 * 0.25) / 0.8)


def test_validate_embeddings_rejects_missing_or_mismatched_values():
    with pytest.raises(ValueError, match="backfill"):
        hybrid_recommender.validate_embeddings(
            [
                MovieEmbedding(
                    movie_id=1,
                    movie_uuid=UUID("00000000-0000-0000-0000-000000000001"),
                    recommendation_profile=None,
                    embedding_model="model",
                    embedding=[1.0],
                )
            ],
            "model",
        )


@pytest.mark.asyncio
async def test_recommend_movies_cf_only_does_not_use_embeddings(monkeypatch):
    async def load_data(_):
        return _movies(), _users(), []

    monkeypatch.setattr(
        hybrid_recommender, "load_data_from_db_for_recommendation", load_data
    )
    monkeypatch.setattr(hybrid_recommender, "load_cf_recommender", lambda *_: object())
    monkeypatch.setattr(
        hybrid_recommender,
        "get_cf_recommendations",
        lambda *args: (
            __import__("torch").tensor([0, 1]),
            __import__("torch").tensor([2.0, 9.0]),
        ),
    )
    monkeypatch.setattr(
        hybrid_recommender,
        "EmbeddingSettings",
        lambda: pytest.fail("CF-only recommendations must not load embeddings"),
    )

    result = await hybrid_recommender.recommend_movies(
        RecommendationInput(username="user", prompt="")
    )

    assert [item.movie.id for item in result] == [2, 1]
    assert [item.match_score for item in result] == [9.0, 2.0]


def test_extract_image_query_uses_image_semantics(monkeypatch):
    class _Extractor:
        def __call__(self, image):
            return type(
                "Semantics",
                (),
                {
                    "description": "rainy neon city",
                    "mood": "melancholic",
                    "energy": "low",
                    "suggested_genres": ["Drama"],
                    "avoid_genres": ["Action"],
                },
            )()

    monkeypatch.setattr(hybrid_recommender, "ImageSemanticExtractor", _Extractor)
    monkeypatch.setattr(hybrid_recommender, "get_lm", lambda: None)

    query = hybrid_recommender.extract_image_query(object())

    assert "rainy neon city" in query
    assert "Mood: melancholic" in query
    assert "Avoid: Action" in query


@pytest.mark.asyncio
async def test_recommend_movies_image_only_embeds_only_image_query(monkeypatch):
    stored = [
        MovieEmbedding(
            movie_id=1,
            movie_uuid=_movies()[0].id_uuid,
            recommendation_profile="one",
            embedding_model="model",
            embedding=[0.0, 1.0],
        ),
        MovieEmbedding(
            movie_id=2,
            movie_uuid=_movies()[1].id_uuid,
            recommendation_profile="two",
            embedding_model="model",
            embedding=[1.0, 0.0],
        ),
    ]

    async def load_data(_):
        return _movies(), _users(), stored

    class _ImageEmbeddingClient:
        def embed(self, texts):
            assert texts == ["rainy neon city"]
            return [[1.0, 0.0]]

    monkeypatch.setattr(
        hybrid_recommender, "load_data_from_db_for_recommendation", load_data
    )
    monkeypatch.setattr(hybrid_recommender, "load_cf_recommender", lambda *_: object())
    monkeypatch.setattr(
        hybrid_recommender,
        "get_cf_recommendations",
        lambda *args: (
            __import__("torch").tensor([0, 1]),
            __import__("torch").tensor([5.0, 5.0]),
        ),
    )
    monkeypatch.setattr(
        hybrid_recommender, "extract_image_query", lambda _: "rainy neon city"
    )
    monkeypatch.setattr(
        hybrid_recommender,
        "EmbeddingSettings",
        lambda: type("S", (), {"MODEL": "model"})(),
    )
    monkeypatch.setattr(hybrid_recommender, "EmbeddingClient", _ImageEmbeddingClient)

    result = await hybrid_recommender.recommend_movies(
        RecommendationInput(
            username="user",
            prompt="",
            image=dspy.Image("https://example.com/image.jpg"),
        )
    )

    assert [item.movie.id for item in result] == [2, 1]


@pytest.mark.asyncio
async def test_backfill_builds_profiles_and_updates_once(monkeypatch):
    movies = _movies()
    updates = []

    class _Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

    class _BackfillEmbeddingClient:
        def embed(self, texts):
            assert texts[0].startswith("One. 2024")
            return [[1.0], [2.0]]

    async def fetch_movies(_):
        return movies

    async def update(_, embeddings):
        updates.append(embeddings)

    monkeypatch.setattr(
        hybrid_recommender, "DatabaseConnector", lambda **_: _Connection()
    )
    monkeypatch.setattr(hybrid_recommender, "DBSettings", lambda: object())
    monkeypatch.setattr(hybrid_recommender, "fetch_movies_from_db", fetch_movies)
    monkeypatch.setattr(hybrid_recommender, "upsert_movie_embeddings", update)
    monkeypatch.setattr(hybrid_recommender, "EmbeddingClient", _BackfillEmbeddingClient)
    monkeypatch.setattr(
        hybrid_recommender,
        "EmbeddingSettings",
        lambda: type("S", (), {"MODEL": "model"})(),
    )

    await hybrid_recommender.backfill_movie_embeddings()

    assert len(updates) == 1
    assert [item.embedding for item in updates[0]] == [[1.0], [2.0]]
    assert all(item.embedding_model == "model" for item in updates[0])
