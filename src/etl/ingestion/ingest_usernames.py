from src.etl.sql_queries.queries import upsert_usernames
from src.settings import WebScraperSettings
from src.etl.generation.generate import generate_usernames


async def ingest_usernames() -> None:
    username_page = WebScraperSettings().USERNAME_PAGE
    usernames = generate_usernames(username_page=username_page)
    await upsert_usernames(usernames)
