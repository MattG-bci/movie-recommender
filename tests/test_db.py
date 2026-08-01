import pytest
import asyncpg

from .conftest import run_sqitch, setup_test_db, load_fixtures
from etl.sql_queries import DatabaseConnector
from settings import DBSettings


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
async def test_sqitch_revert(db_service: DBSettings):
    run_sqitch(db_service, "revert")

    async with DatabaseConnector(db_settings=db_service) as conn:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        table_names = [t["tablename"] for t in tables]
        assert "users" not in table_names
        assert "movies" not in table_names
        assert "movie_ratings" not in table_names

    # Redeploy and reload fixtures so other tests are unaffected
    setup_test_db(db_service)
    load_fixtures(db_service)
