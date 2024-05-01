from src.data_processing.web_scraping import UserScraper, RatingScraper
from src.data_processing.infer_db_pd import transform_data
from src.data_processing.infer_db_pd import load_db, append_data_samples, save_db
from config import config


username_page: str = "https://letterboxd.com/members/popular/this/week/"
####
usr_scraper = UserScraper()
usernames = usr_scraper.request_data(username_page)
####
df = load_db()
for idx, usr in enumerate(usernames):
    print(f"User #{idx}: {usr}")
    target_page: str = f"https://letterboxd.com/{usr}/films/"
    movie_scraper = RatingScraper()
    movie_ratings = movie_scraper.request_data(target_page)
    data = transform_data(usr, movie_ratings)
    df = append_data_samples(data, df)
save_db(df)
