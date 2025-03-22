from schemas.users import UserIn
from src.etl.generation.web_scraping import UserScraper, RatingScraper
from src.settings import WebScraperSettings
import asyncio


def generate_usernames(username_page: str) -> list[UserIn]:
    usr_scraper = UserScraper(username_page_url=username_page)
    usernames = usr_scraper.scrape_pages(n_pages=1)
    return usernames


async def generate_movie_ratings(
    usernames: list[UserIn],
) -> dict[UserIn, list[tuple[str, float]]]:
    rating_scraper = RatingScraper(usernames=usernames)
    movie_data = await rating_scraper.scrape_data()
    return movie_data


if __name__ == "__main__":
    print("Generating username data")
    username_page = WebScraperSettings().USERNAME_PAGE
    usernames = generate_usernames(username_page=username_page)
    print(usernames)
    print("Generating movie data")
    movies = asyncio.run(generate_movie_ratings(usernames=usernames))
    print(movies)
