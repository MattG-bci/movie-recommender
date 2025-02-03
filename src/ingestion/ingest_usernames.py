from src.generation.web_scraping import UserScraper
from src.settings import USERNAME_PAGE, DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT
from src.schemas.users import User
import asyncpg
import asyncio


async def ingest_usernames() -> None:
    usr_scraper = UserScraper(username_page=USERNAME_PAGE)
    usernames = usr_scraper.scrape_pages(start_page=1, end_page=2)

    conn = await asyncpg.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT)

    # TODO: Write a correct SQL query to insert the usernames into the database
    for username in usernames:
        await conn.execute("INSERT INTO users (username) VALUES ($1)", username)




if __name__ == "__main__":
    asyncio.run(ingest_usernames())
