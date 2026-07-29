import pytest

from etl.generation.web_scraping import UserScraper
from schemas.users import User
from datetime import datetime


def test_get_usernames_for_page(fake_site_url):
    scraper = UserScraper(
        username_page_url=f"{fake_site_url}/members/popular/this/week/"
    )
    url = f"{fake_site_url}/members/popular/this/week/page/1"
    usernames = scraper.get_usernames_for_page(url)

    assert len(usernames) == 3
    assert "testuser1" in usernames
    assert "testuser2" in usernames
    assert "testuser3" in usernames


@pytest.mark.asyncio
async def test_scrape_page_incremental(fake_site_url):
    scraper = UserScraper(
        username_page_url=f"{fake_site_url}/members/popular/this/week/"
    )
    existing_usernames = []
    users = await scraper.scrape_page_incremental(existing_usernames)

    assert users is not None
    assert len(users) == 3
    assert {u.username for u in users} == {"testuser1", "testuser2", "testuser3"}


@pytest.mark.asyncio
async def test_scrape_page_incremental_filters_existing(fake_site_url):
    existing_usernames = [
        User(
            id=1,
            username="testuser1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    ]

    scraper = UserScraper(
        username_page_url=f"{fake_site_url}/members/popular/this/week/"
    )
    users = await scraper.scrape_page_incremental(existing_usernames)

    assert users is not None
    usernames = {u.username for u in users}
    assert "testuser1" not in usernames
    assert "testuser2" in usernames
    assert "testuser3" in usernames
