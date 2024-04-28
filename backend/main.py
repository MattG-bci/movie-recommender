from src.data_processing.web_scraping import UserScraper, RatingScraper


target_page: str = "https://letterboxd.com/mattstouche/films/"
username_page: str = "https://letterboxd.com/members/popular/this/week/"
####
usr_scraper = UserScraper()
usernames = usr_scraper.request_data(username_page)
print(usernames)
####
movie_scraper = RatingScraper()
movie_ratings = movie_scraper.request_data(target_page)
print(movie_ratings)
