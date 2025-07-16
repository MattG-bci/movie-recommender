from schemas.movies import MovieRatingIn
from schemas.users import UserIn, User
from etl.generation.web_scraping import UserScraper, RatingScraper


def generate_usernames(username_page: str) -> list[UserIn]:
    usr_scraper = UserScraper(username_page_url=username_page)
    usernames = usr_scraper.scrape_pages(n_pages=1)
    return usernames


async def generate_movie_ratings(
    usernames: list[User],
) -> list[MovieRatingIn]:
    rating_scraper = RatingScraper(usernames=usernames)
    movie_data = await rating_scraper.scrape_data()
    return movie_data
