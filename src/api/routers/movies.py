from fastapi import APIRouter, HTTPException, Query

from api.poster_service import PosterBudget, PosterCache, resolve_posters
from etl.sql_queries import (
    DatabaseConnector,
    count_movies,
    fetch_movie_by_id,
    fetch_movies_page,
)
from schemas.movie import MovieOut, MoviePage
from settings import OMDbSettings

movies = APIRouter()

_omdb_settings = OMDbSettings()
_poster_cache = PosterCache(_omdb_settings.CACHE_PATH)
_poster_budget = PosterBudget(_omdb_settings.DAILY_BUDGET)


@movies.get("")
async def get_movies(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: str | None = None,
) -> MoviePage:
    async with DatabaseConnector() as conn:
        rows = await fetch_movies_page(conn, limit=limit, offset=offset, search=search)
        total = await count_movies(conn, search=search)

    poster_urls = await resolve_posters(
        rows, _omdb_settings, _poster_cache, _poster_budget
    )
    items = [
        MovieOut(**row.model_dump(), poster_url=poster_urls.get(row.id)) for row in rows
    ]
    return MoviePage(items=items, total=total, limit=limit, offset=offset)


@movies.get("/{movie_id}")
async def get_movie(movie_id: int) -> MovieOut:
    async with DatabaseConnector() as conn:
        row = await fetch_movie_by_id(conn, movie_id)

    if row is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    poster_urls = await resolve_posters(
        [row], _omdb_settings, _poster_cache, _poster_budget
    )
    return MovieOut(**row.model_dump(), poster_url=poster_urls.get(row.id))
