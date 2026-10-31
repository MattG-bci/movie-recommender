from etl.generation.generate import generate_movie_ratings
from etl.ingestion import add_uuid_to_movie_ratings
from etl.sql_queries import (
    DatabaseConnector,
    fetch_ratings,
    fetch_user_by_username,
    upsert_movie_ratings,
)
from schemas.movie import MovieRating
from fastapi import APIRouter, HTTPException
import logging


ratings = APIRouter()

logging.basicConfig(level=logging.INFO)


@ratings.get("")
async def get_ratings(
    username: str | None = None, movie_name: str | None = None, limit: int = 100
) -> list[MovieRating]:
    async with DatabaseConnector() as conn:
        return await fetch_ratings(conn, username, movie_name, limit)


@ratings.post("/{username}")
async def insert_ratings_for_user(username: str) -> None:
    async with DatabaseConnector() as conn:
        user = await fetch_user_by_username(conn, username)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    async with DatabaseConnector() as conn:
        movie_ratings = await generate_movie_ratings(conn, usernames=[user])
        movie_ratings_with_uuid = add_uuid_to_movie_ratings(movie_ratings)
        await upsert_movie_ratings(conn, movie_ratings_with_uuid)
