from etl.sql_queries import fetch_usernames_from_db, fetch_movies_from_db
from schemas.movie import MovieRatingIn, MovieIn
from schemas.users import UserIn, User
from etl.generation.web_scraping import UserScraper, RatingScraper, MovieScraper
import logging


logger = logging.getLogger(__name__)


async def generate_usernames(username_page: str) -> list[UserIn]:
    usr_scraper = UserScraper(username_page_url=username_page)
    usernames = await usr_scraper.scrape_page_incremental()
    return usernames


async def generate_movies(movies_page: str) -> list[MovieIn]:
    movie_scraper = MovieScraper(movie_page_url=movies_page)
    existing_movies = await fetch_movies_from_db()
    existing_movies = [movie.title for movie in existing_movies]
    movies = await movie_scraper.get_data_incremental(existing_movies)
    return movies


async def generate_movie_ratings(
    usernames: list[User],
) -> list[MovieRatingIn]:
    usernames_from_db = await fetch_usernames_from_db()
    map_usernames_to_ids = {user.username: user.id for user in usernames_from_db}

    movies_from_db = await fetch_movies_from_db()
    map_movie_titles_to_ids = {movie.title: movie.id for movie in movies_from_db}

    rating_scraper = RatingScraper(usernames=usernames)
    movie_data = await rating_scraper.scrape_data()

    # Filter movie ratings of movies not recorded in the database
    filtered_ratings = list(
        filter(lambda x: x.movie_name in map_movie_titles_to_ids.keys(), movie_data)
    )
    logger.info(f"Filtered ratings: {len(movie_data)} -> {len(filtered_ratings)}")
    movie_data = [
        MovieRatingIn(
            user_id=map_usernames_to_ids[rating.username],
            movie_id=map_movie_titles_to_ids[rating.movie_name],
            rating=rating.rating,
        )
        for rating in filtered_ratings
    ]

    return movie_data
