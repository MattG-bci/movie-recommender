from src.settings import USERNAME_PAGE
from src.generation.generate import generate_usernames
import asyncpg
from src.sql_queries.queries import inject_db_connection


@inject_db_connection
async def ingest_usernames(conn: asyncpg.Connection) -> None:
    usernames = generate_usernames(username_page=USERNAME_PAGE)

    for username in usernames:
        await conn.execute(
            """
                INSERT INTO users (username) VALUES ($1)
                ON CONFLICT (username) DO NOTHING
            """,
            username,
        )
