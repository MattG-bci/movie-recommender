import asyncio
import datetime
import json
import logging
import os
import tempfile

import httpx

from schemas.movie import Movie
from settings import OMDbSettings

logger = logging.getLogger(__name__)


async def fetch_poster_url(
    client: httpx.AsyncClient, title: str, year: int, settings: OMDbSettings
) -> str | None:
    """Look up a single poster URL from OMDb. Never raises: every failure mode
    (not found, no artwork, bad key, quota exhausted, timeout, connection error)
    resolves to None so a single flaky lookup cannot take the endpoint down."""
    try:
        response = await client.get(
            settings.BASE_URL,
            params={"apikey": settings.API_KEY, "t": title, "y": year},
            timeout=settings.TIMEOUT_SECONDS,
        )
    except (httpx.TimeoutException, httpx.RequestError):
        return None

    if response.status_code == 401:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    if payload.get("Response") == "False":
        return None
    if payload.get("Error") == "Request limit reached!":
        return None

    poster = payload.get("Poster")
    if not poster or poster == "N/A":
        return None
    return poster


class PosterCache:
    """(title, year) -> poster url, persisted to disk as JSON so it survives
    API restarts. Negative results (None) are cached too, since re-querying
    films OMDb does not have would silently drain the daily budget."""

    def __init__(self, cache_path) -> None:
        self._cache_path = cache_path
        self._data: dict[str, str | None] = {}
        self.load()

    @staticmethod
    def _key(title: str, year: int) -> str:
        return f"{title}:::{year}"

    def get(self, title: str, year: int) -> str | None:
        return self._data.get(self._key(title, year))

    def contains(self, title: str, year: int) -> bool:
        return self._key(title, year) in self._data

    def set(self, title: str, year: int, url: str | None) -> None:
        self._data[self._key(title, year)] = url

    def load(self) -> None:
        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Malformed poster cache: expected a JSON object")
            self._data = data
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            self._data = {}

    def flush(self) -> None:
        directory = os.path.dirname(str(self._cache_path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp_path, self._cache_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


class PosterBudget:
    """UTC-date-scoped daily request counter."""

    def __init__(self, daily_budget: int) -> None:
        self._daily_budget = daily_budget
        self._date = datetime.datetime.now(datetime.timezone.utc).date()
        self._consumed = 0

    def _maybe_reset(self) -> None:
        today = datetime.datetime.now(datetime.timezone.utc).date()
        if today != self._date:
            self._date = today
            self._consumed = 0

    def try_consume(self, n: int) -> int:
        self._maybe_reset()
        remaining = max(self._daily_budget - self._consumed, 0)
        granted = min(n, remaining)
        self._consumed += granted
        return granted


async def resolve_posters(
    movies: list[Movie],
    settings: OMDbSettings,
    cache: PosterCache,
    budget: PosterBudget,
    client: httpx.AsyncClient | None = None,
) -> dict[int, str | None]:
    """Movie id -> poster URL. Cache hits are free. Uncached films are looked
    up in page order under the daily budget; anything beyond the granted
    allowance resolves to None and renders as a fallback tile."""
    if not settings.API_KEY:
        return {movie.id: None for movie in movies}

    results: dict[int, str | None] = {}
    uncached: list[Movie] = []
    for movie in movies:
        if cache.contains(movie.title, movie.release_year):
            results[movie.id] = cache.get(movie.title, movie.release_year)
        else:
            uncached.append(movie)

    granted = budget.try_consume(len(uncached))
    to_fetch, to_skip = uncached[:granted], uncached[granted:]
    for movie in to_skip:
        results[movie.id] = None

    if to_fetch:
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENCY)

        async def _run(active_client: httpx.AsyncClient) -> None:
            async def _fetch(movie: Movie) -> tuple[Movie, str | None]:
                async with semaphore:
                    url = await fetch_poster_url(
                        active_client, movie.title, movie.release_year, settings
                    )
                return movie, url

            fetched = await asyncio.gather(*(_fetch(movie) for movie in to_fetch))
            for movie, url in fetched:
                results[movie.id] = url
                cache.set(movie.title, movie.release_year, url)

        if client is not None:
            await _run(client)
        else:
            async with httpx.AsyncClient() as owned_client:
                await _run(owned_client)

        cache.flush()

    return results
