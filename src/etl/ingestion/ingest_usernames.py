from etl.sql_queries.queries import upsert_usernames
from settings import WebScraperSettings
from etl.generation.generate import generate_usernames


async def ingest_usernames() -> None:
    username_page = WebScraperSettings().USERNAME_PAGE
    usernames = await generate_usernames(username_page=username_page)
    await upsert_usernames(usernames)
