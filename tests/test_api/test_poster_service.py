import asyncio

import httpx
import pytest

from api.poster_service import (
    PosterBudget,
    PosterCache,
    fetch_poster_url,
    resolve_posters,
)
from schemas.movie import Movie
from settings import OMDbSettings


def _make_settings(**overrides) -> OMDbSettings:
    defaults = dict(
        API_KEY="test-key",
        BASE_URL="https://www.omdbapi.com/",
        TIMEOUT_SECONDS=3.0,
        MAX_CONCURRENCY=8,
        DAILY_BUDGET=900,
        CACHE_PATH="unused-in-test.json",
    )
    defaults.update(overrides)
    return OMDbSettings(**defaults)


def _make_movie(
    id: int = 1, title: str = "Some Movie", release_year: int = 2020
) -> Movie:
    return Movie(
        id=id,
        title=title,
        release_year=release_year,
        genres=["genre1"],
        director="director1",
        country="country1",
        actors=["actor1"],
    )


def _client_with_handler(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_fetch_poster_url__returns_url_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"Response": "True", "Poster": "http://x/p.jpg"}
        )

    async with _client_with_handler(handler) as client:
        result = await fetch_poster_url(client, "Some Movie", 2020, _make_settings())

    assert result == "http://x/p.jpg"


@pytest.mark.asyncio
async def test_fetch_poster_url__returns_none_when_response_is_false():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"Response": "False", "Error": "Movie not found!"}
        )

    async with _client_with_handler(handler) as client:
        result = await fetch_poster_url(client, "Unknown", 2020, _make_settings())

    assert result is None


@pytest.mark.asyncio
async def test_fetch_poster_url__returns_none_when_poster_is_na():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"Response": "True", "Poster": "N/A"})

    async with _client_with_handler(handler) as client:
        result = await fetch_poster_url(client, "Some Movie", 2020, _make_settings())

    assert result is None


@pytest.mark.asyncio
async def test_fetch_poster_url__returns_none_on_invalid_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"Response": "False", "Error": "Invalid API key!"}
        )

    async with _client_with_handler(handler) as client:
        result = await fetch_poster_url(client, "Some Movie", 2020, _make_settings())

    assert result is None


@pytest.mark.asyncio
async def test_fetch_poster_url__returns_none_when_quota_exceeded():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"Response": "False", "Error": "Request limit reached!"}
        )

    async with _client_with_handler(handler) as client:
        result = await fetch_poster_url(client, "Some Movie", 2020, _make_settings())

    assert result is None


@pytest.mark.asyncio
async def test_fetch_poster_url__returns_none_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    async with _client_with_handler(handler) as client:
        result = await fetch_poster_url(client, "Some Movie", 2020, _make_settings())

    assert result is None


@pytest.mark.asyncio
async def test_fetch_poster_url__returns_none_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client_with_handler(handler) as client:
        result = await fetch_poster_url(client, "Some Movie", 2020, _make_settings())

    assert result is None


def test_poster_cache__caches_negative_results(tmp_path):
    cache = PosterCache(tmp_path / "cache.json")
    cache.set("Some Movie", 2020, None)

    assert cache.get("Some Movie", 2020) is None
    assert cache.contains("Some Movie", 2020) is True


def test_poster_cache__contains_distinguishes_none_from_missing(tmp_path):
    cache = PosterCache(tmp_path / "cache.json")

    assert cache.contains("Untouched", 2020) is False
    cache.set("Untouched", 2020, None)
    assert cache.contains("Untouched", 2020) is True


def test_poster_cache__persists_across_instances(tmp_path):
    path = tmp_path / "cache.json"
    cache = PosterCache(path)
    cache.set("Some Movie", 2020, "http://x/p.jpg")
    cache.flush()

    reloaded = PosterCache(path)
    assert reloaded.get("Some Movie", 2020) == "http://x/p.jpg"


def test_poster_cache__missing_file_is_treated_as_empty(tmp_path):
    cache = PosterCache(tmp_path / "does-not-exist.json")
    assert cache.contains("Some Movie", 2020) is False


def test_poster_cache__malformed_file_is_treated_as_empty(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("not json{{{")

    cache = PosterCache(path)
    assert cache.contains("Some Movie", 2020) is False


def test_poster_budget__grants_up_to_remaining_allowance():
    budget = PosterBudget(daily_budget=5)
    assert budget.try_consume(3) == 3
    assert budget.try_consume(4) == 2


def test_poster_budget__returns_zero_when_exhausted():
    budget = PosterBudget(daily_budget=2)
    budget.try_consume(2)
    assert budget.try_consume(1) == 0


def test_poster_budget__resets_on_new_utc_date(monkeypatch):
    import datetime

    budget = PosterBudget(daily_budget=2)
    budget.try_consume(2)
    assert budget.try_consume(1) == 0

    tomorrow = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)

    class _FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return tomorrow

    monkeypatch.setattr("api.poster_service.datetime.datetime", _FakeDateTime)
    assert budget.try_consume(1) == 1


@pytest.mark.asyncio
async def test_resolve_posters__returns_all_none_when_api_key_missing(tmp_path):
    settings = _make_settings(API_KEY=None)
    cache = PosterCache(tmp_path / "cache.json")
    budget = PosterBudget(daily_budget=900)
    movies = [_make_movie(1), _make_movie(2, title="Other Movie")]

    result = await resolve_posters(movies, settings, cache, budget)

    assert result == {1: None, 2: None}


@pytest.mark.asyncio
async def test_resolve_posters__does_not_refetch_cached_titles(tmp_path):
    settings = _make_settings()
    cache = PosterCache(tmp_path / "cache.json")
    cache.set("Some Movie", 2020, "http://cached/p.jpg")
    budget = PosterBudget(daily_budget=900)

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"Response": "False"})

    async with _client_with_handler(handler) as client:
        result = await resolve_posters(
            [_make_movie(1)], settings, cache, budget, client=client
        )

    assert result == {1: "http://cached/p.jpg"}
    assert calls == []


@pytest.mark.asyncio
async def test_resolve_posters__cache_hits_do_not_consume_budget(tmp_path):
    settings = _make_settings()
    cache = PosterCache(tmp_path / "cache.json")
    cache.set("Some Movie", 2020, "http://cached/p.jpg")
    budget = PosterBudget(daily_budget=900)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"Response": "False"})

    async with _client_with_handler(handler) as client:
        await resolve_posters([_make_movie(1)], settings, cache, budget, client=client)

    assert budget.try_consume(900) == 900


@pytest.mark.asyncio
async def test_resolve_posters__fills_in_page_order_when_budget_partial(tmp_path):
    settings = _make_settings()
    cache = PosterCache(tmp_path / "cache.json")
    budget = PosterBudget(daily_budget=1)
    movies = [_make_movie(1, title="First"), _make_movie(2, title="Second")]

    def handler(request: httpx.Request) -> httpx.Response:
        title = request.url.params.get("t")
        return httpx.Response(
            200, json={"Response": "True", "Poster": f"http://x/{title}.jpg"}
        )

    async with _client_with_handler(handler) as client:
        result = await resolve_posters(movies, settings, cache, budget, client=client)

    assert result[1] == "http://x/First.jpg"
    assert result[2] is None


@pytest.mark.asyncio
async def test_resolve_posters__respects_max_concurrency(tmp_path):
    settings = _make_settings(MAX_CONCURRENCY=2)
    cache = PosterCache(tmp_path / "cache.json")
    budget = PosterBudget(daily_budget=900)
    movies = [_make_movie(i, title=f"Movie{i}") for i in range(6)]

    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal concurrent, max_concurrent
        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent -= 1
        title = request.url.params.get("t")
        return httpx.Response(
            200, json={"Response": "True", "Poster": f"http://x/{title}.jpg"}
        )

    async with _client_with_handler(handler) as client:
        await resolve_posters(movies, settings, cache, budget, client=client)

    assert max_concurrent <= 2
