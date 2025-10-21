from bs4 import BeautifulSoup

from httpx import Response
from pydantic import BaseModel
from requests import get
from requests.exceptions import RequestException

import httpx
import asyncio
from tenacity import retry, wait_exponential
from playwright.async_api import async_playwright

import os
import itertools

from etl.sql_queries.queries import fetch_usernames_from_db, fetch_movies_from_db
from schemas.movie import MovieRatingIn, MovieIn
from schemas.users import UserIn, User
from settings import WebScraperSettings
import logging

logger = logging.getLogger(__name__)


class UserScraper(BaseModel):
    username_page_url: str

    async def scrape_page_incremental(self) -> list[UserIn]:
        existing_usernames = await fetch_usernames_from_db()
        existing_usernames = [user.username for user in existing_usernames]
        page = 1
        while True:
            username_url = os.path.join(self.username_page_url, "page", str(page))
            try:
                fetched_usernames = self.get_usernames_for_page(username_url)
            except Exception as e:
                logger.error(f"Scraping failed on page {page}. Error: {e}")
                break
            new_usernames = list(set(fetched_usernames) - set(existing_usernames))
            if new_usernames:
                logger.info(f"Fetched {len(new_usernames)} new usernames from page {page}.")
                usernames = [UserIn(username=username) for username in new_usernames]
                return usernames
            logger.info(f"No new usernames found on page {page}. Scraping the next page...")
            page += 1

    def get_usernames_for_page(self, username_url: str) -> list[str]:
        response = self.request_data(username_url)
        soup = BeautifulSoup(response.content, features="html.parser")
        usernames = self.get_data(soup)
        return usernames

    @staticmethod
    def request_data(usernames_page: str) -> Response:
        try:
            username_response = get(usernames_page)
        except RequestException:
            raise RequestException("Error in the request to the usernames page.")
        return username_response

    @staticmethod
    def get_data(soup: BeautifulSoup) -> list[str]:
        usernames = []
        users = soup.find_all("div", class_="person-summary")
        for user in users:
            user = user.find("a", class_="name")["href"].split("/")[1]
            usernames.append(user)
        return usernames


class RatingScraper(BaseModel):
    usernames: list[User]
    map_username_to_id: dict[str, int]
    map_movie_to_id: dict[str, int]

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def request_data(self, target_page: str) -> httpx.Response:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                response = await client.get(target_page)
                response.raise_for_status()
            except httpx.RequestError as exc:
                raise Exception(f"Error in the request to {target_page}.") from exc
        return response

    async def scrape_data(self) -> list[MovieRatingIn]:
        tasks = [
            self.scrape_data_per_username(username.username)
            for username in self.usernames
        ]
        results = await asyncio.gather(*tasks)

        movie_ratings = []
        for user, user_ratings in zip(self.usernames, results):
            logger.info(f"Fetched {len(user_ratings)} ratings for user: {user.username}")
            for title, movie_rating in user_ratings:
                user_id = self.map_username_to_id.get(user.username)
                movie_id = self.map_movie_to_id.get(title)
                if movie_id is None:
                    logger.info(f"Movie '{title}' not found in the database. Skipping rating for user '{user.username}'.")
                    continue
                movie_ratings.append(MovieRatingIn(user_id=user_id, movie_id=movie_id, rating=movie_rating))

        logger.info(f"Scraping completed. Total movie ratings fetched: {len(movie_ratings)} for {len(self.usernames)} users.")
        return movie_ratings

    async def scrape_data_per_username(self, username: str) -> list[tuple[str, float]]:
        logger.info(f"Scraping ratings for user: {username}")
        target_page = os.path.join(WebScraperSettings().RATINGS_PAGE, username, "films")
        response = await self.request_data(target_page)
        soup = BeautifulSoup(response.content, features="html.parser")

        try:
            pages_div = soup.find("div", class_="paginate-pages")
            n_pages = int(pages_div.find_all("li", class_="paginate-page")[-1].text)
        except AttributeError:
            n_pages = 1

        tasks = [
            self.get_data(os.path.join(target_page, "page", str(page)))
            for page in range(1, n_pages + 1)
        ]
        all_movie_data = await asyncio.gather(*tasks)
        return list(itertools.chain(*all_movie_data))

    async def get_data(self, target_url: str) -> list[tuple[str, float]]:
        response = await self.request_data(target_url)
        soup = BeautifulSoup(response.content, features="html.parser")
        return self.scrape_movie_ratings(soup)

    def scrape_movie_ratings(
        self, soup: BeautifulSoup
    ) -> list[tuple[str, float]]:
        movie_ratings = []
        poster_containers = soup.find_all("li", class_="griditem")
        for poster in poster_containers:
            movie_title = poster.find("img")["alt"]
            rating = poster.find("span", class_="rating")
            if not rating:
                continue
            num_rating = self.convert_rating(rating.text)
            movie_ratings.append((movie_title, num_rating))
        return movie_ratings

    @staticmethod
    def convert_rating(rating: str) -> float:
        num_rating = float(len(rating)) if rating[-1] != "½" else len(rating) - 0.5
        return num_rating * 2


class MovieScraper(BaseModel):
    movie_page_url: str

    async def get_data_incremental(self) -> list[MovieIn]:
        existing_movies = await fetch_movies_from_db()
        existing_movie_titles = {movie.title for movie in existing_movies}

        n_page = 1
        while True:
            target_page = os.path.join(self.movie_page_url, str(n_page))
            resp = await self.request_data(target_page)
            soup = BeautifulSoup(resp, features="html.parser")
            try:
                new_movies = await self.scrape_movies(soup, existing_movie_titles)
            except Exception as e:
                logger.error(f"Error while scraping page {n_page}: {e}")
                break
            if new_movies:
                return new_movies
            logger.info(f"No new movies found on page {n_page}. Scraping next page...")
            n_page += 1

    async def scrape_movies(self, soup: BeautifulSoup, existing_movie_titles: set[str]) -> list[MovieIn]:
        movies = []
        movie_containers = soup.find_all("div", class_="poster film-poster")
        for movie in movie_containers:
            movie_information = movie.find("a", class_="frame")
            data = movie_information.text.split(" ")
            title = " ".join(data[:-1])
            if title in existing_movie_titles:
                logger.info(f"Movie '{title}' already exists in the database. Skipping...")
                continue
            release_year = data[-1]
            release_year = int(release_year.strip("()"))

            movie_link = "https://letterboxd.com" + movie_information["href"]
            resp = await self.request_data(movie_link)
            local_soup = BeautifulSoup(resp, features="html.parser")
            actors = [actor.text for actor in local_soup.find("div", class_="cast-list").find_all("a")[:5]] # Only get first 5 actors
            director = local_soup.find("a", class_="contributor").text
            country = local_soup.find("div", id="tab-details").find_all("div", class_="text-sluglist")
            country = [row.find("a") for row in country]
            country = [row.text for row in country if "country" in row["href"]][0].strip(" ")
            genres = local_soup.find("div", id="tab-genres").find_next("div", class_="text-sluglist capitalize").find_all("a")
            genres = [genre.text for genre in genres]

            movies.append(
                MovieIn(title=title, release_year=release_year, director=director, actors=actors, genres=genres, country=country)
            )
        return movies

    @staticmethod
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def request_data(url: str) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            html = await page.content()
            await browser.close()
        return html
