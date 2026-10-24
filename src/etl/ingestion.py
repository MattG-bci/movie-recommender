from etl.generation.generate import (
    create_uuid_for_movie,
    create_uuid_for_movie_rating,
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
from schemas.movie import MovieIn, MovieWithUUID, MovieRatingIn, MovieRatingWithUUID
from settings import WebScraperSettings


def add_uuid_to_movies(movies: list[MovieIn]) -> list[MovieWithUUID]:
    return [
        MovieWithUUID(
            id=create_uuid_for_movie(movie.title, movie.release_year, movie.director),
            title=movie.title,
            release_year=movie.release_year,
            director=movie.director,
            country=movie.country,
            actors=movie.actors,
            genres=movie.genres,
            description=movie.description,
        )
        for movie in movies
    ]


def add_uuid_to_movie_ratings(
    movie_ratings: list[MovieRatingIn],
) -> list[MovieRatingWithUUID]:
    return [
        MovieRatingWithUUID(
            id=create_uuid_for_movie_rating(rating.user_id, rating.movie_id),
            user_id=rating.user_id,
            movie_id=rating.movie_id,
            rating=rating.rating,
        )
        for rating in movie_ratings
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
        movie_ratings_with_uuid = add_uuid_to_movie_ratings(movie_ratings)
        await upsert_movie_ratings(conn, movie_ratings_with_uuid)
