from datetime import datetime

import pytest

from etl.generation.web_scraping import RatingScraper
from schemas.users import User


def make_user(username: str, user_id: int = 1) -> User:
    return User(
        id=user_id,
        username=username,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.mark.asyncio
async def test_scrape_single_user(fake_site_url):
    user = make_user("testuser1", user_id=1)
    scraper = RatingScraper(usernames=[user])

    ratings = await scraper.scrape_data()

    assert len(ratings) == 3
    titles = [r.movie_name for r in ratings]
    assert "The Matrix" in titles
    assert "Inception" in titles
    assert "Pulp Fiction" in titles


@pytest.mark.asyncio
async def test_rating_values(fake_site_url):
    user = make_user("testuser1", user_id=1)
    scraper = RatingScraper(usernames=[user])

    ratings = await scraper.scrape_data()
    rating_map = {r.movie_name: r.rating for r in ratings}

    assert rating_map["The Matrix"] == 10.0
    assert rating_map["Inception"] == 8.0
    assert rating_map["Pulp Fiction"] == 7.0


@pytest.mark.asyncio
async def test_scrape_multiple_users(fake_site_url):
    users = [
        make_user("testuser1", user_id=1),
        make_user("testuser2", user_id=2),
    ]
    scraper = RatingScraper(usernames=users)

    ratings = await scraper.scrape_data()

    user1_ratings = [r for r in ratings if r.username == "testuser1"]
    user2_ratings = [r for r in ratings if r.username == "testuser2"]

    assert len(user1_ratings) == 3
    assert len(user2_ratings) == 2


@pytest.mark.asyncio
async def test_skips_unrated_movies(fake_site_url):
    user = make_user("testuser2", user_id=2)
    scraper = RatingScraper(usernames=[user])

    ratings = await scraper.scrape_data()
    titles = [r.movie_name for r in ratings]

    assert "No Rating Movie" not in titles


def test_convert_rating():
    assert RatingScraper.convert_rating("★★★★★") == 10.0
    assert RatingScraper.convert_rating("★★★★") == 8.0
    assert RatingScraper.convert_rating("★★★") == 6.0
    assert RatingScraper.convert_rating("★★") == 4.0
    assert RatingScraper.convert_rating("★") == 2.0
    assert RatingScraper.convert_rating("★★★★½") == 9.0
    assert RatingScraper.convert_rating("★★★½") == 7.0
    assert RatingScraper.convert_rating("★★½") == 5.0
    assert RatingScraper.convert_rating("★½") == 3.0
    assert RatingScraper.convert_rating("½") == 1.0
