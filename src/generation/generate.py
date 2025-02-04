from .web_scraping import UserScraper
from src.schemas.users import User


def generate_usernames(username_page: str) -> list[str]:
    usr_scraper = UserScraper(username_page=username_page)
    usernames = usr_scraper.scrape_pages(start_page=1, end_page=11)
    return usernames
