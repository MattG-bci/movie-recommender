from schemas.users import UserIn
from etl.generation.generate import generate_movie_ratings
from etl.sql_queries.queries import fetch_usernames_from_db
import asyncpg


async def ingest_movie_ratings(usernames: UserIn[str]) -> None:
    usernames = await fetch_usernames_from_db()
    movie_ratings = generate_movie_ratings(usernames=usernames)
    conn = await asyncpg.connect(
    )
