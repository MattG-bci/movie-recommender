from etl.sql_queries.queries import fetch_usernames_from_db, fetch_movies_from_db
from schemas.movie import MovieRatingIn
from schemas.users import UserIn, User
from etl.generation.web_scraping import UserScraper, RatingScraper

async def generate_usernames(username_page: str) -> list[UserIn]:
    usr_scraper = UserScraper(username_page_url=username_page)
    usernames = await usr_scraper.scrape_page_incremental()
    return usernames


async def generate_movie_ratings(
    usernames: list[User],
) -> list[MovieRatingIn]:
    usernames_from_db = await fetch_usernames_from_db()
    map_usernames_to_ids = {user.username: user.id for user in usernames_from_db}

    movies_from_db = await fetch_movies_from_db()
    map_movie_titles_to_ids = {movie.title: movie.id for movie in movies_from_db}

    rating_scraper = RatingScraper(usernames=usernames, map_username_to_id=map_usernames_to_ids, map_movie_to_id=map_movie_titles_to_ids)
    movie_data = await rating_scraper.scrape_data()
    return movie_data
