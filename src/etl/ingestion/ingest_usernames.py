from sql_queries.queries import upsert_usernames
from etl import USERNAME_PAGE
from etl import generate_usernames


async def ingest_usernames() -> None:
    usernames = generate_usernames(username_page=USERNAME_PAGE)
    await upsert_usernames(usernames)
