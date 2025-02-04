from src.generation.web_scraping import UserScraper
from src.settings import USERNAME_PAGE, DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT
from src.schemas.users import User
from src.generation.generate import generate_usernames
import asyncpg
import asyncio


async def ingest_usernames() -> None:
    usernames = generate_usernames(username_page=USERNAME_PAGE)

    conn = await asyncpg.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT)

    for username in usernames:
        await conn.execute(
            """
                INSERT INTO users (username) VALUES ($1)
                ON CONFLICT (username) DO NOTHING
            """, username)




if __name__ == "__main__":
    asyncio.run(ingest_usernames())
