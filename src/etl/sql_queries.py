import logging

import asyncpg
from pydantic import BaseModel

from typing import Any, Callable, Coroutine

from schemas.movie import MovieRatingIn, Movie, MovieRatingWithId
from schemas.users import UserIn, User, UserProfile
from settings import DBSettings


class DatabaseConnector(BaseModel):
    connection: asyncpg.Connection = None
    db_settings: DBSettings = DBSettings()

    model_config = dict(arbitrary_types_allowed=True)

    async def __aenter__(self) -> asyncpg.Connection:
        self.connection = await asyncpg.connect(
            host=self.db_settings.HOST,
            user=self.db_settings.USER,
            password=self.db_settings.PASS,
            database=self.db_settings.NAME,
            port=self.db_settings.PORT,
        )
        return self.connection

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.connection.close()


def inject_db_connection(func) -> Callable:
    async def inner_wrapper(*args, **kwargs) -> Coroutine:
        async with DatabaseConnector() as conn:
            res = await func(conn, *args, **kwargs)
        return res

    return inner_wrapper


@inject_db_connection
async def insert_usernames(conn: asyncpg.Connection, usernames: list[UserIn]) -> None:
    logging.info(f"Inserting {len(usernames)} usernames to the database...")
    if not usernames:
        logging.info("No new usernames to upsert.")
        return

    query = f"""
        INSERT INTO users ({", ".join(list(usernames[0].model_dump().keys()))}) VALUES ($1)
    """

    params = [list(data.model_dump().values()) for data in usernames]
    await conn.executemany(
        query,
        params,
    )


async def upsert_movie_ratings(movie_ratings: list[MovieRatingIn]) -> None:
    logging.info(f"Upserting {len(movie_ratings)} movie ratings to the database...")
    await upsert_to_db(
        data_to_upsert=movie_ratings,
        table_name="movie_ratings",
        conflict_columns=["user_id", "movie_id"],
    )


@inject_db_connection
async def fetch_usernames_from_db(conn: asyncpg.Connection) -> list[User]:
    query = "SELECT * FROM users"
    rows = await conn.fetch(query)
    return [User(**dict(row)) for row in rows]


@inject_db_connection
async def fetch_movies_from_db(conn: asyncpg.Connection) -> list[Movie]:
    query = (
        "SELECT id, title, release_year, genres, director, country, actors FROM movies"
    )
    rows = await conn.fetch(query)
    return [Movie(**dict(row)) for row in rows]


@inject_db_connection
async def fetch_movie_ratings_from_db(
    conn: asyncpg.Connection,
) -> list[MovieRatingWithId]:
    query = "SELECT id, user_id, movie_id, rating FROM movie_ratings"
    rows = await conn.fetch(query)
    return [MovieRatingWithId(**dict(row)) for row in rows]


@inject_db_connection
async def fetch_movie_ratings_from_db_for_movie(
    conn: asyncpg.Connection,
    movie_id: int,
    limit: int,
) -> list[MovieRatingWithId]:
    query = f"SELECT id, user_id, movie_id, rating FROM movie_ratings WHERE movie_id = {movie_id} LIMIT {limit}"
    rows = await conn.fetch(query)
    return [MovieRatingWithId(**dict(row)) for row in rows]


@inject_db_connection
async def fetch_user_profile(
    conn: asyncpg.Connection, user_id: int, top_k: int = 5
) -> UserProfile:
    query = f"""

    WITH top_actors AS (
        SELECT
            UNNEST(mv.actors) AS actor,
            AVG(mv_rat.rating) AS average_rating
        FROM users u
        JOIN movie_ratings mv_rat ON u.id = mv_rat.user_id
        JOIN movies mv ON mv.id = mv_rat.movie_id
        WHERE u.id = {user_id}
        GROUP BY actor
        ORDER BY average_rating DESC
        LIMIT {top_k}
    ),
    top_genres AS (
        SELECT
            UNNEST(mv.genres) AS genre,
            AVG(mv_rat.rating) AS average_rating
        FROM users u
        JOIN movie_ratings mv_rat ON u.id = mv_rat.user_id
        JOIN movies mv ON mv.id = mv_rat.movie_id
        WHERE u.id = {user_id}
        GROUP BY genre
        ORDER BY average_rating DESC
        LIMIT {top_k}
    ),
    top_directors AS (
        SELECT
            mv.director,
            AVG(mv_rat.rating) AS average_rating
        FROM users u
        JOIN movie_ratings mv_rat ON u.id = mv_rat.user_id
        JOIN movies mv ON mv.id = mv_rat.movie_id
        WHERE u.id = {user_id}
        GROUP BY director
        ORDER BY average_rating DESC
        LIMIT {top_k}
    ),
    top_movies AS (
        SELECT
            u.id,
            u.username,
            mv.title,
            mv_rat.rating
        FROM users u
        JOIN movie_ratings mv_rat ON u.id = mv_rat.user_id
        JOIN movies mv ON mv.id = mv_rat.movie_id
        WHERE u.id = {user_id}
        ORDER BY mv_rat.rating DESC
        LIMIT {top_k}
    )
    SELECT
        (SELECT UNNEST(ARRAY_AGG(distinct id)) from top_movies) as user_id,
	    (SELECT UNNEST(ARRAY_AGG(distinct username)) from top_movies) as username,
        (SELECT ARRAY_AGG(actor ORDER BY average_rating DESC) FROM top_actors) AS top_actors,
        (SELECT ARRAY_AGG(director ORDER BY average_rating DESC) FROM top_directors) AS top_directors,
        (SELECT ARRAY_AGG(genre ORDER BY average_rating DESC) FROM top_genres) AS top_genres,
        (SELECT ARRAY_AGG(title ORDER BY rating DESC) FROM top_movies) AS top_movies;
    """

    row = await conn.fetch(query)
    return UserProfile(**dict(row[0]))


@inject_db_connection
async def upsert_to_db(
    conn: asyncpg.Connection,
    data_to_upsert: list[BaseModel],
    table_name: str,
    conflict_columns: list[str] = ("id",),
) -> None:
    if not data_to_upsert:
        logging.info("No data to upsert.")
        return

    # Get column names
    column_names = list(data_to_upsert[0].model_dump().keys())

    # Create the correct number of placeholders
    placeholders = ", ".join(f"${i + 1}" for i in range(len(column_names)))

    # Create SET clause for updates
    set_clause = ", ".join(
        f"{col} = excluded.{col}" for col in column_names if col not in conflict_columns
    )

    query = f"""
        INSERT INTO {table_name} ({", ".join(column_names)})
        VALUES ({placeholders})
        ON CONFLICT ({", ".join(conflict_columns)})
        DO UPDATE SET {set_clause}
    """

    params = [list(data.model_dump().values()) for data in data_to_upsert]
    await conn.executemany(query, params)
