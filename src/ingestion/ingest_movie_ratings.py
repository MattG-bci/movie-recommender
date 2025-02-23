from src.settings import DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT
from src.generation.generate import generate_movie_ratings
import asyncpg


async def ingest_movie_ratings(usernames: list[str]) -> None:
    movie_ratings = generate_movie_ratings(usernames=usernames)


    conn = await asyncpg.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT
    )
