from etl.generation.generate import (
    generate_usernames,
    generate_movie_ratings,
    generate_movies,
)
from etl.sql_queries import (
    upsert_to_db,
    insert_usernames,
    fetch_usernames_from_db,
    upsert_movie_ratings,
)
from settings import WebScraperSettings


async def ingest_movies() -> None:
    movie_page_url = WebScraperSettings().MOVIES_PAGE
    movies = await generate_movies(movies_page=movie_page_url)
    await upsert_to_db(
        movies, "movies", conflict_columns=["title", "release_year", "director"]
    )


async def ingest_usernames() -> None:
    username_page = WebScraperSettings().USERNAME_PAGE
    usernames = await generate_usernames(username_page=username_page)
    await insert_usernames(usernames=usernames)


async def ingest_movie_ratings() -> None:
    usernames = await fetch_usernames_from_db()
    movie_ratings = await generate_movie_ratings(usernames=usernames[:2])
    await upsert_movie_ratings(movie_ratings)
