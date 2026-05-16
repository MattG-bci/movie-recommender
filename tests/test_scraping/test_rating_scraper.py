import os
from unittest.mock import MagicMock

import pytest
from bs4 import BeautifulSoup

from etl.generation.web_scraping import RatingScraper


def _load_fixture(path: str) -> str:
    with open(os.path.join(path, "index.html")) as f:
        return f.read()


def _make_rating_scraper(base_url: str, **kwargs) -> RatingScraper:
    defaults = dict(
        usernames=[],
        map_username_to_id={},
        map_movie_to_id={},
        base_url=base_url,
    )
    defaults.update(kwargs)
    return RatingScraper(**defaults)


class TestConvertRating:
    @pytest.mark.parametrize(
        "stars, expected",
        [
            ("★", 2.0),
            ("★★", 4.0),
            ("★★★", 6.0),
            ("★★★★", 8.0),
            ("★★★★★", 10.0),
            ("★★★½", 7.0),
            ("½", 1.0),
        ],
    )
    def test_converts_star_ratings(self, stars, expected):
        assert RatingScraper.convert_rating(stars) == expected


class TestScrapeMovieRatings:
    def test_parses_ratings_from_html(self, fixtures_path):
        html = _load_fixture(
            os.path.join(fixtures_path, "testuser1", "films", "page", "1")
        )
        soup = BeautifulSoup(html, features="html.parser")
        scraper = _make_rating_scraper(fixtures_path)
        ratings = scraper.scrape_movie_ratings(soup)
        assert ratings == [("The Godfather", 10.0), ("Pulp Fiction", 7.0)]

    def test_skips_unrated_movies(self, fixtures_path):
        html = _load_fixture(
            os.path.join(fixtures_path, "testuser1", "films", "page", "1")
        )
        soup = BeautifulSoup(html, features="html.parser")
        scraper = _make_rating_scraper(fixtures_path)
        ratings = scraper.scrape_movie_ratings(soup)
        titles = [title for title, _ in ratings]
        assert "Unrated Movie" not in titles


class TestScrapeDataPerUsername:
    @pytest.mark.asyncio
    async def test_scrapes_all_pages_for_user(self, fixtures_path, monkeypatch):
        scraper = _make_rating_scraper(fixtures_path)

        async def mock_request_data(self, url):
            html = _load_fixture(url)
            response = MagicMock()
            response.content = html.encode()
            return response

        monkeypatch.setattr(RatingScraper, "request_data", mock_request_data)

        ratings = await scraper.scrape_data_per_username("testuser1")
        assert ("The Godfather", 10.0) in ratings
        assert ("Pulp Fiction", 7.0) in ratings
        assert ("Inception", 8.0) in ratings
        assert len(ratings) == 3
