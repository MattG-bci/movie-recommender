from etl.sql_queries import DatabaseConnector, fetch_ratings
from schemas.movie import MovieRating
from fastapi import APIRouter


ratings = APIRouter()


@ratings.get("")
async def get_ratings(
    username: str | None = None, movie_name: str | None = None, limit: int = 100
) -> list[MovieRating]:
    async with DatabaseConnector() as conn:
        return await fetch_ratings(conn, username, movie_name, limit)
