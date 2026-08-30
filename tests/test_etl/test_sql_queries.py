import pytest

from etl.sql_queries import (
    DatabaseConnector,
    fetch_usernames_from_db,
    fetch_user_profile,
)
from schemas.recommendation import UserProfile


@pytest.mark.asyncio
async def test_fetch_usernames_from_db(db_service):
    async with DatabaseConnector(db_settings=db_service) as conn:
        out = await fetch_usernames_from_db(conn)

    out_list = [row.username for row in out]
    expected = ["testuser1", "testuser2", "testuser3"]
    assert out_list == expected


@pytest.mark.asyncio
async def test_fetch_user_profile(db_service):
    mock_username = "testuser1"
    async with DatabaseConnector(db_settings=db_service) as conn:
        out = await fetch_user_profile(conn, mock_username)

    expected = UserProfile(
        top_genres=["genre3", "genre4", "genre2", "genre1"],
        top_actors=["actor8", "actor2", "actor1"],
        top_directors=["director2", "director8", "director3", "director1"],
        top_movies=["test_movie2", "test_movie8", "test_movie3", "test_movie1"],
    )
    assert out == expected
