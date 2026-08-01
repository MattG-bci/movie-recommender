import pytest

from etl.sql_queries import DatabaseConnector, fetch_usernames_from_db


@pytest.mark.asyncio
async def test_fetch_usernames_from_db(db_service):
    async with DatabaseConnector(db_settings=db_service) as conn:
        out = await fetch_usernames_from_db(conn)

    out_list = [row.username for row in out]
    expected = ["testuser1", "testuser2", "testuser3"]
    assert out_list == expected
