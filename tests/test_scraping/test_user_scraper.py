import os
from unittest.mock import MagicMock

from bs4 import BeautifulSoup

from etl.generation.web_scraping import UserScraper


def _load_fixture(path: str) -> str:
    with open(os.path.join(path, "index.html")) as f:
        return f.read()


class TestGetData:
    def test_parses_usernames_from_html(self, fixtures_path):
        html = _load_fixture(
            os.path.join(
                fixtures_path, "members", "popular", "this", "week", "page", "1"
            )
        )
        soup = BeautifulSoup(html, features="html.parser")
        usernames = UserScraper.get_data(soup)
        assert usernames == ["testuser1", "testuser2"]

    def test_returns_empty_list_for_empty_page(self, fixtures_path):
        html = _load_fixture(
            os.path.join(
                fixtures_path, "members", "popular", "this", "week", "page", "2"
            )
        )
        soup = BeautifulSoup(html, features="html.parser")
        usernames = UserScraper.get_data(soup)
        assert usernames == []


class TestGetUsernamesForPage:
    def test_fetches_and_parses_usernames(self, fixtures_path, monkeypatch):
        base_url = os.path.join(fixtures_path, "members", "popular", "this", "week")
        scraper = UserScraper(base_url=base_url)

        page_url = os.path.join(base_url, "page", "1")
        response = MagicMock()
        response.content = _load_fixture(page_url)
        monkeypatch.setattr(
            UserScraper, "request_data", staticmethod(lambda url: response)
        )

        usernames = scraper.get_usernames_for_page(page_url)
        assert usernames == ["testuser1", "testuser2"]
