import logging
from datetime import datetime

import asyncpg
from pydantic import BaseModel

from typing import Any, Callable, Coroutine

from schemas.users import UserIn
from src.settings import DBSettings


class DatabaseConnector(BaseModel):
    connection: asyncpg.Connection = None
    db_settings: DBSettings = DBSettings()

    model_config = dict(arbitrary_types_allowed=True)

    async def __aenter__(self) -> asyncpg.Connection:
        self.connection = await asyncpg.connect(
            host=self.db_settings.HOST,
            user=self.db_settings.USER,
            password=self.db_settings.PASS,
            database=self.db_settings.NAME,
            port=self.db_settings.PORT,
        )
        return self.connection

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.connection.close()


def inject_db_connection(func) -> Callable:
    async def inner_wrapper(*args, **kwargs) -> Coroutine:
        async with DatabaseConnector() as conn:
            res = await func(conn, *args, **kwargs)
        return res

    return inner_wrapper


async def upsert_usernames(usernames: list[UserIn]) -> None:
    logging.info(f"Upserting {len(usernames)} usernames to the database...")

    await upsert_to_db(usernames, "users")


@inject_db_connection
async def upsert_to_db(
    conn: asyncpg.Connection, data_to_upsert: list[BaseModel], table_name: str
) -> None:
    now = datetime.now()

    query = f"""
        INSERT INTO {table_name} ({" ,".join(list(data_to_upsert[0].dict().keys()))}) VALUES ($1)
        ON CONFLICT (username) DO UPDATE SET updated_at = $2
    """

    await conn.executemany(
        query, [[*data.dict().values(), now] for data in data_to_upsert]
    )
