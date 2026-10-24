from uuid import NAMESPACE_DNS, uuid4, uuid5

import pytest
import pytest_asyncio

from etl.sql_queries import (
    DatabaseConnector,
    count_movies,
    fetch_movie_by_id,
    fetch_movie_embeddings,
    fetch_movie_ratings_from_db,
    fetch_movie_ratings_from_db_for_movie,
    fetch_movies_from_db,
    fetch_movies_page,
    fetch_ratings,
    fetch_user_by_username,
    fetch_user_profile,
    fetch_usernames_from_db,
    insert_usernames,
    upsert_movie_embeddings,
    upsert_movie_ratings,
    upsert_to_db,
)
from schemas.movie import MovieRating, MovieRatingWithUUID, MovieWithUUID
from schemas.recommendation import MovieEmbedding, UserProfile
from schemas.users import UserWithUUID


def fixture_user_id(username: str):
    return uuid5(NAMESPACE_DNS, username)


def fixture_movie_id(title: str, release_year: int, director: str):
    return uuid5(NAMESPACE_DNS, f"{title}{release_year}{director}")


@pytest_asyncio.fixture
async def db_connection(db_service):
    """Provide a real Postgres connection and roll back test mutations."""
    async with DatabaseConnector(db_settings=db_service) as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            yield conn
        finally:
            await transaction.rollback()


@pytest.mark.asyncio
async def test_insert_usernames_inserts_users(db_connection):
    user = UserWithUUID(id=uuid4(), username=f"insert-user-{uuid4().hex}")

    await insert_usernames(db_connection, [user])

    row = await db_connection.fetchrow(
        "SELECT id, username FROM users WHERE id = $1", user.id
    )
    assert dict(row) == {"id": user.id, "username": user.username}


@pytest.mark.asyncio
async def test_upsert_movie_ratings_updates_existing_rating(db_connection):
    user_id = fixture_user_id("testuser1")
    movie_id = fixture_movie_id("test_movie1", 2024, "director1")
    rating = MovieRatingWithUUID(
        id=uuid4(), user_id=user_id, movie_id=movie_id, rating=9
    )

    await upsert_movie_ratings(db_connection, [rating])

    row = await db_connection.fetchrow(
        "SELECT id, rating FROM movie_ratings WHERE user_id = $1 AND movie_id = $2",
        user_id,
        movie_id,
    )
    assert dict(row) == {"id": rating.id, "rating": 9}


@pytest.mark.asyncio
async def test_fetch_ratings_filters_and_limits_fixture_ratings(db_connection):
    ratings = await fetch_ratings(db_connection, "testuser1", "test_movie2", 1)

    assert ratings == [
        MovieRating(username="testuser1", movie_name="test_movie2", rating=10)
    ]


@pytest.mark.asyncio
async def test_fetch_usernames_from_db(db_connection):
    users = await fetch_usernames_from_db(db_connection)

    assert {user.username for user in users} == {
        "testuser1",
        "testuser2",
        "testuser3",
    }


@pytest.mark.asyncio
async def test_fetch_movies_from_db_returns_fixture_movies(db_connection):
    movies = await fetch_movies_from_db(db_connection)

    assert {movie.title for movie in movies} == {
        f"test_movie{number}" for number in range(1, 9)
    }


@pytest.mark.asyncio
async def test_fetch_movie_embeddings_returns_movies_without_embeddings(db_connection):
    embeddings = await fetch_movie_embeddings(db_connection)

    assert len(embeddings) == 8
    assert all(embedding.recommendation_profile is None for embedding in embeddings)
    assert all(embedding.embedding_model is None for embedding in embeddings)
    assert all(embedding.embedding is None for embedding in embeddings)


@pytest.mark.asyncio
async def test_upsert_movie_embeddings_inserts_and_updates_embedding(db_connection):
    movie_id = fixture_movie_id("test_movie1", 2024, "director1")
    initial = MovieEmbedding(
        movie_id=movie_id,
        recommendation_profile="initial profile",
        embedding_model="test-model-v1",
        embedding=[0.1, 0.2],
    )
    updated = initial.model_copy(
        update={
            "recommendation_profile": "updated profile",
            "embedding_model": "test-model-v2",
            "embedding": [0.3, 0.4],
        }
    )

    await upsert_movie_embeddings(db_connection, [initial])
    await upsert_movie_embeddings(db_connection, [updated])

    row = await db_connection.fetchrow(
        """
        SELECT recommendation_profile, embedding_model, embedding
        FROM movie_embeddings
        WHERE movie_id = $1
        """,
        movie_id,
    )
    assert dict(row) == {
        "recommendation_profile": "updated profile",
        "embedding_model": "test-model-v2",
        "embedding": [0.3, 0.4],
    }


@pytest.mark.asyncio
async def test_fetch_movies_page_filters_orders_and_paginates(db_connection):
    page = await fetch_movies_page(db_connection, limit=2, offset=2)
    matching = await fetch_movies_page(db_connection, search="movie1")

    assert [movie.title for movie in page] == ["test_movie3", "test_movie4"]
    assert [movie.title for movie in matching] == ["test_movie1"]


@pytest.mark.asyncio
async def test_fetch_movie_by_id_returns_matching_movie_or_none(db_connection):
    movie_id = fixture_movie_id("test_movie1", 2024, "director1")

    movie = await fetch_movie_by_id(db_connection, movie_id)
    missing = await fetch_movie_by_id(db_connection, uuid4())

    assert movie is not None
    assert movie.title == "test_movie1"
    assert missing is None


@pytest.mark.asyncio
async def test_count_movies_filters_fixture_movies(db_connection):
    assert await count_movies(db_connection) == 8
    assert await count_movies(db_connection, search="movie1") == 1


@pytest.mark.asyncio
async def test_fetch_user_by_username_returns_matching_user_or_none(db_connection):
    user = await fetch_user_by_username(db_connection, "testuser2")
    missing = await fetch_user_by_username(db_connection, "unknown-user")

    assert user is not None
    assert user.id == fixture_user_id("testuser2")
    assert missing is None


@pytest.mark.asyncio
async def test_fetch_movie_ratings_from_db_returns_fixture_ratings(db_connection):
    ratings = await fetch_movie_ratings_from_db(db_connection)

    assert len(ratings) == 14
    assert {(rating.user_id, rating.movie_id, rating.rating) for rating in ratings} >= {
        (
            fixture_user_id("testuser1"),
            fixture_movie_id("test_movie2", 1970, "director2"),
            10,
        )
    }


@pytest.mark.asyncio
async def test_fetch_movie_ratings_from_db_for_movie_applies_limit(db_connection):
    movie_id = fixture_movie_id("test_movie7", 1929, "director7")

    ratings = await fetch_movie_ratings_from_db_for_movie(db_connection, movie_id, 1)

    assert len(ratings) == 1
    assert ratings[0].movie_id == movie_id


@pytest.mark.asyncio
async def test_fetch_user_profile(db_connection):
    profile = await fetch_user_profile(db_connection, "testuser1")

    assert profile == UserProfile(
        top_genres=["genre3", "genre4", "genre2", "genre1"],
        top_actors=["actor8", "actor2", "actor1"],
        top_directors=["director2", "director8", "director3", "director1"],
        top_movies=["test_movie2", "test_movie8", "test_movie3", "test_movie1"],
    )


@pytest.mark.asyncio
async def test_upsert_to_db_inserts_and_updates_a_movie(db_connection):
    movie = MovieWithUUID(
        id=uuid4(),
        title=f"upsert-movie-{uuid4().hex}",
        release_year=2026,
        director="Test Director",
        country="Test Country",
        actors=["Test Actor"],
        genres=["Test Genre"],
        description="original description",
    )

    await upsert_to_db(db_connection, [movie], "movies")
    await upsert_to_db(
        db_connection,
        [movie.model_copy(update={"description": "updated description"})],
        "movies",
    )

    row = await db_connection.fetchrow(
        "SELECT title, description FROM movies WHERE id = $1", movie.id
    )
    assert dict(row) == {
        "title": movie.title,
        "description": "updated description",
    }
