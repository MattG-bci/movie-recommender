import typer

from etl.ingestion import ingest_movies, ingest_usernames, ingest_movie_ratings
import asyncio
from functools import wraps
import logging

from etl.sql_queries import fetch_movie_ratings_from_db
from model.recommender import train_movie_recommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
app = typer.Typer(no_args_is_help=True)


def async_typer_command(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@app.command()
@async_typer_command
async def ingest_users() -> None:
    await ingest_usernames()


@app.command()
@async_typer_command
async def ingest_movies_command() -> None:
    await ingest_movies()


@app.command()
@async_typer_command
async def ingest_ratings() -> None:
    await ingest_movie_ratings()


@app.command()
@async_typer_command
async def run_all_ingestion() -> None:
    await ingest_usernames()
    await ingest_movies()
    await ingest_movie_ratings()


@app.command()
@async_typer_command
async def train_recommender() -> None:
    ratings = await fetch_movie_ratings_from_db()
    train_movie_recommender(ratings)


if __name__ == "__main__":
    app()
