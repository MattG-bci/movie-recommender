"""
Cache HTML from Letterboxd endpoints for use as test fixtures.

Usage:
    python tests/scraper/cache_html.py

This script fetches real HTML from Letterboxd and saves it to
tests/fixtures/html/ mirroring the URL path structure. Run this
manually to refresh fixtures with real-world HTML.

The scrapers can then be tested against an nginx container
that serves these cached files.
"""

import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://letterboxd.com"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Paths to cache. Extend this list as needed.
# Each entry is a URL path relative to the Letterboxd base URL.
PATHS_TO_CACHE = [
    "/members/popular/this/week/page/1",
]


def fetch_and_save(url_path: str) -> bool:
    url = BASE_URL + url_path
    logger.info(f"Fetching {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return False

    file_path = FIXTURES_DIR / url_path.lstrip("/") / "index.html"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(response.text, encoding="utf-8")
    logger.info(f"Saved to {file_path}")
    return True


def discover_and_cache_user_pages(usernames: list[str]) -> None:
    """Fetch rating pages for a list of usernames."""
    for username in usernames:
        path = f"/{username}/films/"
        fetch_and_save(path)
        time.sleep(1)


def discover_and_cache_movie_pages(movie_slugs: list[str]) -> None:
    """Fetch detail pages for a list of movie slugs."""
    for slug in movie_slugs:
        path = f"/film/{slug}/"
        fetch_and_save(path)
        time.sleep(1)


def main() -> None:
    logger.info(f"Caching HTML to {FIXTURES_DIR}")

    for path in PATHS_TO_CACHE:
        if not fetch_and_save(path):
            logger.warning(f"Skipping {path}")
        time.sleep(1)

    # Cache the popular films list page
    fetch_and_save("/films/popular/page/1")
    time.sleep(1)

    # To cache user-specific and movie-specific pages, pass usernames/slugs:
    #   discover_and_cache_user_pages(["username1", "username2"])
    #   discover_and_cache_movie_pages(["the-matrix", "inception"])
    #
    # Or parse the already-cached members page to extract usernames,
    # then the popular films page to extract movie slugs, and fetch those.

    logger.info("Done. Review and commit the cached fixtures.")


if __name__ == "__main__":
    main()
