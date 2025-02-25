from sql_queries.queries import upsert_usernames
from src.settings import USERNAME_PAGE
from src.generation.generate import generate_usernames


async def ingest_usernames() -> None:
    usernames = generate_usernames(username_page=USERNAME_PAGE)
    await upsert_usernames(usernames)
