import pytest
import asyncpg


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
