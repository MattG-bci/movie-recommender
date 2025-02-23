
import asyncpg
from pydantic import BaseModel

from typing import Any, Callable, Coroutine

from src.settings import DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT


class DatabaseConnector(BaseModel):
    conn: asyncpg.Connection = None
    host: str
    user: str
    password: str
    database: str
    port: int

    model_config = dict(arbitrary_types_allowed=True)

    async def __aenter__(self) -> asyncpg.Connection:
        self.conn = await asyncpg.connect(host=self.host, user=self.user, password=self.password, database=self.database, port=self.port)
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.conn.close()


def inject_db_connection(func) -> Callable:
    async def inner_wrapper(*args, **kwargs) -> Coroutine:
        async with DatabaseConnector(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT) as conn:
            res = await func(conn, *args, **kwargs)
        return res
    return inner_wrapper


def upsert_to_db():
    pass
