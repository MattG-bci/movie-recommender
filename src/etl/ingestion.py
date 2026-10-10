import uuid

from etl.generation.generate import (
    generate_usernames,
    generate_movie_ratings,
    generate_movies,
)
from etl.sql_queries import (
    DatabaseConnector,
    upsert_to_db,
    insert_usernames,
    fetch_usernames_from_db,
    upsert_movie_ratings,
)
from schemas.movie import MovieIn, MovieWithUUID
from settings import WebScraperSettings
from uuid import uuid5, UUID


def create_uuid_for_movies(title: str, release_year: int, director: str) -> UUID:
    return uuid5(uuid.NAMESPACE_DNS, title + str(release_year) + director)


def add_uuid_to_movies(movies: list[MovieIn]) -> list[MovieWithUUID]:
    return [
        MovieWithUUID(
            id_uuid=create_uuid_for_movies(
                movie.title, movie.release_year, movie.director
            ),
            title=movie.title,
            release_year=movie.release_year,
            director=movie.director,
            country=movie.country,
            actors=movie.actors,
            genres=movie.genres,
        )
        for movie in movies
    ]


async def ingest_movies() -> None:
    movie_page_url = WebScraperSettings().MOVIES_PAGE
    async with DatabaseConnector() as conn:
        movies = await generate_movies(conn, movies_page=movie_page_url)
        movies_with_uuid = add_uuid_to_movies(movies)
        await upsert_to_db(
            conn,
            movies_with_uuid,
            "movies",
            conflict_columns=["title", "release_year", "director"],
        )


async def ingest_usernames() -> None:
    username_page = WebScraperSettings().USERNAME_PAGE
    async with DatabaseConnector() as conn:
        usernames = await generate_usernames(conn, username_page=username_page)
        await insert_usernames(conn, usernames=usernames)


async def ingest_movie_ratings() -> None:
    async with DatabaseConnector() as conn:
        usernames = await fetch_usernames_from_db(conn)
        movie_ratings = await generate_movie_ratings(conn, usernames=usernames)
        await upsert_movie_ratings(conn, movie_ratings)
