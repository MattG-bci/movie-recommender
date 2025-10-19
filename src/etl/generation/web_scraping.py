from bs4 import BeautifulSoup

from httpx import Response, AsyncClient
from pydantic import BaseModel
from requests import get
from requests.exceptions import RequestException

import httpx
import asyncio
from tenacity import retry, wait_exponential

import os
import itertools

from etl.sql_queries.queries import fetch_usernames_from_db
from schemas.movies import MovieRatingIn, MovieIn
from schemas.users import UserIn, User
from settings import WebScraperSettings
import logging

logger = logging.getLogger(__name__)


class UserScraper(BaseModel):
    username_page_url: str

    async def scrape_page(self) -> list[UserIn]:
        usernames = []
        existing_usernames = await fetch_usernames_from_db()
        existing_usernames = [user.username for user in existing_usernames]
        page = 1
        while page <= 10000:
            username_url = os.path.join(self.username_page_url, "page", str(page))
            fetched_usernames = self.get_usernames_for_page(username_url)
            new_usernames = list(set(fetched_usernames) - set(existing_usernames))
            if not new_usernames:
                page += 1
                continue
            logger.info(f"Fetched {len(new_usernames)} new usernames from page {page}.")
            usernames.extend(new_usernames)
            break
        usernames = [UserIn(username=username) for username in usernames]
        return usernames

    def get_usernames_for_page(self, username_url: str) -> list[str]:
        response = self.request_data(username_url)
        soup = BeautifulSoup(response.content, features="html.parser")
        usernames = self.get_data(soup)
        return self.remove_duplicates(usernames)

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

    @staticmethod
    def remove_duplicates(usernames: list[str]) -> list[str]:
        return list(set(usernames))


class RatingScraper(BaseModel):
    usernames: list[User]

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
                movie_ratings.append(MovieRatingIn(user=user.username, movie=title, rating=movie_rating))

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

from playwright.async_api import async_playwright

class MovieScraper(BaseModel):
    movie_page_url: str

    async def get_data(self) -> list[MovieIn]:
        resp = await self.fetch_dynamic_html(self.movie_page_url)
        soup = BeautifulSoup(resp, features="html.parser")
        movies = await self.scrape_movies(soup)
        return movies

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def request_data(self, target_page: str) -> httpx.Response:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json",
            "Referer": target_page,
            "X-Requested-With": "XMLHttpRequest"
        }
        async with AsyncClient(follow_redirects=True) as client:
            try:
                response = await client.get(target_page, headers=headers)
                response.raise_for_status()
            except httpx.RequestError as exc:
                raise Exception(f"Error in the request to {target_page}.") from exc
        return response

    async def scrape_movies(self, soup: BeautifulSoup) -> list[MovieIn]:
        movies = []
        movie_containers = soup.find_all("div", class_="poster film-poster")
        for movie in movie_containers:
            data = movie.find("a", class_="frame")
            title, release_year = data.text.split(" ")
            release_year = int(release_year.strip("()"))

            movie_link = "https://letterboxd.com" + data["href"]
            resp = await self.fetch_dynamic_html(movie_link)
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
            break
        return movies

    @staticmethod
    async def fetch_dynamic_html(url: str) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            html = await page.content()
            await browser.close()
        return html
