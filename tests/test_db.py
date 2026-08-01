import pytest
import asyncpg

from .conftest import run_sqitch, setup_test_db
from etl.sql_queries import DatabaseConnector, fetch_usernames_from_db
from settings import DBSettings

REVERT_DB_NAME = "test_revert"


@pytest.mark.asyncio
async def test_db_connection(db_service):
    conn = await asyncpg.connection.connect(
        user=db_service.USER,
        password=db_service.PASS,
        database=db_service.NAME,
        host=db_service.HOST,
        port=db_service.PORT,
        timeout=5,
    )
    assert conn is not None
    await conn.close()


@pytest.mark.asyncio
async def test_fetch_usernames_from_db(db_service: DBSettings):
    async with DatabaseConnector(db_settings=db_service) as conn:
        users = await fetch_usernames_from_db(conn=conn)

    assert len(users) == 3
    assert [u.username for u in users] == ["testuser1", "testuser2", "testuser3"]


@pytest.mark.asyncio
async def test_sqitch_revert(db_service: DBSettings):
    admin_settings = db_service.model_copy()
    admin_settings.NAME = "postgres"

    revert_settings = db_service.model_copy()
    revert_settings.NAME = REVERT_DB_NAME

    # Create isolated database
    async with DatabaseConnector(db_settings=admin_settings) as conn:
        await conn.execute(f"DROP DATABASE IF EXISTS {REVERT_DB_NAME}")
        await conn.execute(f"CREATE DATABASE {REVERT_DB_NAME}")

    try:
        setup_test_db(revert_settings)
        run_sqitch(revert_settings, "revert")

        async with DatabaseConnector(db_settings=revert_settings) as conn:
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )

        table_names = [t["tablename"] for t in tables]
        assert "users" not in table_names
        assert "movies" not in table_names
        assert "movie_ratings" not in table_names
    finally:
        async with DatabaseConnector(db_settings=admin_settings) as conn:
            await conn.execute(f"DROP DATABASE IF EXISTS {REVERT_DB_NAME}")
