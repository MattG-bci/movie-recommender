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
from settings import WebScraperSettings


async def ingest_movies() -> None:
    movie_page_url = WebScraperSettings().MOVIES_PAGE
    async with DatabaseConnector() as conn:
        movies = await generate_movies(conn, movies_page=movie_page_url)
        await upsert_to_db(
            conn,
            movies,
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
