from src.generation.web_scraping import UserScraper, RatingScraper
from settings import USERNAME_PAGE
import asyncio


def generate_usernames(username_page: str) -> list[str]:
    usr_scraper = UserScraper(username_page_url=username_page)
    usernames = usr_scraper.scrape_pages(n_pages=2)
    return usernames


def generate_movie_ratings(
    username_urls: list[str],
) -> dict[str, list[tuple[str, float]]]:
    rating_scraper = RatingScraper(username_urls=username_urls)
    movie_data = asyncio.run(rating_scraper.scrape_data())
    return movie_data


if __name__ == "__main__":
    import logging

    logging.info("Generating username data")
    usernames = generate_usernames(username_page=USERNAME_PAGE)
    print(usernames)
    logging.info("Generating movie data")
    movies = generate_movie_ratings(username_urls=usernames)
    print(movies)
