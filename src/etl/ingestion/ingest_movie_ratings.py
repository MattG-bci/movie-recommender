from schemas.users import UserIn
from etl.generation.generate import generate_movie_ratings
import asyncpg


async def ingest_movie_ratings(usernames: UserIn[str]) -> None:
    movie_ratings = generate_movie_ratings(usernames=usernames)

    conn = await asyncpg.connect(
    )
