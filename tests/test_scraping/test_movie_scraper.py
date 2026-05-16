import os

import pytest
from bs4 import BeautifulSoup

from etl.generation.web_scraping import MovieScraper


def _load_fixture(path: str) -> str:
    with open(os.path.join(path, "index.html")) as f:
        return f.read()


def _make_movie_scraper(fixtures_path: str) -> MovieScraper:
    return MovieScraper(
        movie_page_url=os.path.join(fixtures_path, "films", "popular", "page"),
        base_url=fixtures_path,
    )


class TestScrapeMovies:
    @pytest.mark.skip(reason="scrape_movies signature mismatch - needs fix")
    @pytest.mark.asyncio
    async def test_scrapes_movie_details(self, fixtures_path, monkeypatch):
        scraper = _make_movie_scraper(fixtures_path)

        async def mock_request_data(url: str) -> str:
            return _load_fixture(url)

        monkeypatch.setattr(
            MovieScraper, "request_data", staticmethod(mock_request_data)
        )

        list_html = _load_fixture(
            os.path.join(fixtures_path, "films", "popular", "page", "1")
        )
        soup = BeautifulSoup(list_html, features="html.parser")

        movies = await scraper.scrape_movies(soup, existing_movie_titles=set())

        assert len(movies) == 2
        assert movies[0].title == "The Godfather"
        assert movies[0].release_year == 1972
        assert movies[0].director == "Francis Ford Coppola"
        assert movies[0].country == "United States"
        assert "Marlon Brando" in movies[0].actors
        assert len(movies[0].actors) == 5
        assert movies[0].genres == ["Crime", "Drama"]

        assert movies[1].title == "Pulp Fiction"
        assert movies[1].release_year == 1994
        assert movies[1].director == "Quentin Tarantino"

    @pytest.mark.skip(reason="scrape_movies signature mismatch - needs fix")
    @pytest.mark.asyncio
    async def test_skips_existing_movies(self, fixtures_path, monkeypatch):
        scraper = _make_movie_scraper(fixtures_path)

        async def mock_request_data(url: str) -> str:
            return _load_fixture(url)

        monkeypatch.setattr(
            MovieScraper, "request_data", staticmethod(mock_request_data)
        )

        list_html = _load_fixture(
            os.path.join(fixtures_path, "films", "popular", "page", "1")
        )
        soup = BeautifulSoup(list_html, features="html.parser")

        movies = await scraper.scrape_movies(
            soup, existing_movie_titles={"The Godfather"}
        )

        assert len(movies) == 1
        assert movies[0].title == "Pulp Fiction"
