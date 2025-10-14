from etl.generation.web_scraping import MovieScraper
from etl.sql_queries.queries import upsert_to_db
from settings import WebScraperSettings
import asyncio

async def ingest_movies() -> None:
    movie_scraper = MovieScraper(movie_page_url=WebScraperSettings().MOVIES_PAGE)
    movies = await movie_scraper.get_data()
    await upsert_to_db(movies, "movies")


if __name__ == "__main__":
    asyncio.run(ingest_movies())
