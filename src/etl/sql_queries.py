import logging

import asyncpg
import psycopg2
from psycopg2.extensions import connection
from pydantic import BaseModel

from typing import Any

from schemas.movie import MovieRatingIn, Movie, MovieRatingWithId, MovieRating
from schemas.users import UserIn, User
from schemas.recommendation import UserProfile
from settings import DBSettings


class DatabaseConnector(BaseModel):
    connection: asyncpg.Connection | connection | None = None
    db_settings: DBSettings | None = None

    model_config = dict(arbitrary_types_allowed=True)

    def _get_settings(self) -> DBSettings:
        return self.db_settings or DBSettings()

    def __enter__(self) -> connection:
        settings = self._get_settings()
        self.connection = psycopg2.connect(
            host=settings.HOST,
            user=settings.USER,
            password=settings.PASS,
            database=settings.NAME,
            port=settings.PORT,
        )
        return self.connection

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.connection.close()

    async def __aenter__(self) -> asyncpg.Connection:
        settings = self._get_settings()
        self.connection = await asyncpg.connect(
            host=settings.HOST,
            user=settings.USER,
            password=settings.PASS,
            database=settings.NAME,
            port=settings.PORT,
        )
        return self.connection

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.connection.close()


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


async def upsert_movie_ratings(
    conn: asyncpg.Connection, movie_ratings: list[MovieRatingIn]
) -> None:
    logging.info(f"Upserting {len(movie_ratings)} movie ratings to the database...")
    await upsert_to_db(
        conn,
        data_to_upsert=movie_ratings,
        table_name="movie_ratings",
        conflict_columns=["user_id", "movie_id"],
    )


async def fetch_ratings(
    conn: asyncpg.Connection, username: str | None, movie_name: str | None, limit: int
) -> list[MovieRating]:
    query = """
    SELECT
        u.username,
        m.title as movie_name,
        mr.rating
    FROM movie_ratings mr
    JOIN users u
    ON u.id = mr.user_id
    JOIN movies m
    ON m.id = mr.movie_id
    WHERE ($1::text IS NULL OR u.username = $1)
    AND ($2::text IS NULL OR m.title = $2)
    LIMIT $3::integer
    ORDER BY u.username, m.title
    """
    rows = await conn.fetch(query, username, movie_name, limit)
    return [MovieRating(**dict(row)) for row in rows]


async def fetch_usernames_from_db(conn: asyncpg.Connection) -> list[User]:
    query = "SELECT * FROM users"
    rows = await conn.fetch(query)
    return [User(**dict(row)) for row in rows]


async def fetch_movies_from_db(conn: asyncpg.Connection) -> list[Movie]:
    query = (
        "SELECT id, title, release_year, genres, director, country, actors FROM movies"
    )
    rows = await conn.fetch(query)
    return [Movie(**dict(row)) for row in rows]


async def fetch_movie_ratings_from_db(
    conn: asyncpg.Connection,
) -> list[MovieRatingWithId]:
    query = "SELECT id, user_id, movie_id, rating FROM movie_ratings"
    rows = await conn.fetch(query)
    return [MovieRatingWithId(**dict(row)) for row in rows]


async def fetch_movie_ratings_from_db_for_movie(
    conn: asyncpg.Connection,
    movie_id: int,
    limit: int,
) -> list[MovieRatingWithId]:
    query = "SELECT id, user_id, movie_id, rating FROM movie_ratings WHERE movie_id = $1 LIMIT $2"
    rows = await conn.fetch(query, movie_id, limit)
    return [MovieRatingWithId(**dict(row)) for row in rows]


async def fetch_user_profile(
    conn: asyncpg.Connection, user_id: int, top_k: int = 5
) -> UserProfile:
    query = """

    WITH top_actors AS (
        SELECT
            UNNEST(mv.actors) AS actor,
            AVG(mv_rat.rating) AS average_rating
        FROM users u
        JOIN movie_ratings mv_rat ON u.id = mv_rat.user_id
        JOIN movies mv ON mv.id = mv_rat.movie_id
        WHERE u.id = $1
        GROUP BY actor
        ORDER BY average_rating DESC
        LIMIT $2
    ),
    top_genres AS (
        SELECT
            UNNEST(mv.genres) AS genre,
            AVG(mv_rat.rating) AS average_rating
        FROM users u
        JOIN movie_ratings mv_rat ON u.id = mv_rat.user_id
        JOIN movies mv ON mv.id = mv_rat.movie_id
        WHERE u.id = $1
        GROUP BY genre
        ORDER BY average_rating DESC
        LIMIT $2
    ),
    top_directors AS (
        SELECT
            mv.director,
            AVG(mv_rat.rating) AS average_rating
        FROM users u
        JOIN movie_ratings mv_rat ON u.id = mv_rat.user_id
        JOIN movies mv ON mv.id = mv_rat.movie_id
        WHERE u.id = $1
        GROUP BY director
        ORDER BY average_rating DESC
        LIMIT $2
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
        WHERE u.id = $1
        ORDER BY mv_rat.rating DESC
        LIMIT $2
    )
    SELECT
        (SELECT UNNEST(ARRAY_AGG(distinct id)) from top_movies) as user_id,
	    (SELECT UNNEST(ARRAY_AGG(distinct username)) from top_movies) as username,
        (SELECT ARRAY_AGG(actor ORDER BY average_rating DESC) FROM top_actors) AS top_actors,
        (SELECT ARRAY_AGG(director ORDER BY average_rating DESC) FROM top_directors) AS top_directors,
        (SELECT ARRAY_AGG(genre ORDER BY average_rating DESC) FROM top_genres) AS top_genres,
        (SELECT ARRAY_AGG(title ORDER BY rating DESC) FROM top_movies) AS top_movies;
    """

    row = await conn.fetchrow(query, user_id, top_k)
    missing = [key for key, value in row.items() if value is None]
    if len(missing) > 0:
        raise ValueError(
            f"Profile for user_id={user_id} has null fields: {" ".join(missing)}"
        )
    return UserProfile(**row)


async def upsert_to_db(
    conn: asyncpg.Connection,
    data_to_upsert: list[BaseModel],
    table_name: str,
    conflict_columns: list[str] = ("id",),
) -> None:
    if not data_to_upsert:
        logging.info("No data to upsert.")
        return

    column_names = list(data_to_upsert[0].model_dump().keys())

    placeholders = ", ".join(f"${i + 1}" for i in range(len(column_names)))

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
