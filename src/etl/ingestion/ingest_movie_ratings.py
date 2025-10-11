from etl.generation.generate import generate_movie_ratings
from etl.sql_queries.queries import fetch_usernames_from_db, upsert_movie_ratings
import asyncio

async def ingest_movie_ratings() -> None:
    usernames = await fetch_usernames_from_db()
    movie_ratings = await generate_movie_ratings(usernames=usernames[:2])
    await upsert_movie_ratings(movie_ratings)


if __name__ == "__main__":
    asyncio.run(ingest_movie_ratings())
