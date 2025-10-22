import asyncpg
import pytest


@pytest.mark.asyncio
async def test_db_connection():
    conn = await asyncpg.connection.connect(
        user="test",
        password="test",
        database="test",
        host="localhost",
        port=5432,
        timeout=5,
    )
    assert conn is not None
    await conn.close()
