from src.generation.web_scraping import UserScraper, RatingScraper
from settings import USERNAME_PAGE
import asyncio


def generate_usernames(username_page: str) -> list[str]:
    usr_scraper = UserScraper(username_page_url=username_page)
    usernames = usr_scraper.scrape_pages(n_pages=1)
    return usernames


async def generate_movie_ratings(
    username_urls: list[str],
) -> dict[str, list[tuple[str, float]]]:
    rating_scraper = RatingScraper(username_urls=username_urls)
    movie_data = await rating_scraper.scrape_data()
    return movie_data


if __name__ == "__main__":
    print("Generating username data")
    usernames = generate_usernames(username_page=USERNAME_PAGE)
    print(usernames)
    print("Generating movie data")
    movies = asyncio.run(generate_movie_ratings(username_urls=usernames))
    print(movies)
