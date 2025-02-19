from bs4 import BeautifulSoup

from abc import ABC, abstractmethod

from httpx import Response
from pydantic import BaseModel
from requests import get
from requests.exceptions import RequestException
from typing import Tuple, List

from settings import RATINGS_PAGE

import os
import logging


class BaseWebScraper(ABC):
    @abstractmethod
    def request_data(self, url: str) -> Response:
        raise NotImplementedError

    @abstractmethod
    def get_data(self, scraper: BeautifulSoup) -> None:
        raise NotImplementedError


class UserScraper(BaseModel, BaseWebScraper):
    username_page_url: str

    def scrape_pages(self, n_pages: int = 10) -> list[str]:
        usernames = []
        for page in range(1, n_pages + 1):
            username_url = os.path.join(self.username_page_url, "page", str(page))
            usernames += self.get_usernames_for_page(username_url)
        return usernames

    def get_usernames_for_page(self, username_url: str) -> list[str]:
        response = self.request_data(username_url)
        soup = BeautifulSoup(response.content, features="html.parser")
        usernames = self.get_data(soup)
        return self.remove_duplicates(usernames)

    def request_data(self, usernames_page: str) -> Response:
        try:
            username_response = get(usernames_page)
        except RequestException:
            raise RequestException("Error in the request to the usernames page.")
        return username_response

    def get_data(self, soup: BeautifulSoup) -> list[str]:
        usernames = []
        users = soup.find_all("div", class_="person-summary")
        for user in users:
            user = user.find("a", class_="name")["href"].split("/")[1]
            usernames.append(user)
        return usernames

    @staticmethod
    def remove_duplicates(usernames: list[str]) -> list[str]:
        return list(set(usernames))


class RatingScraper(BaseModel, BaseWebScraper):
    username_urls: list[str]

    async def request_data(self, target_page: str) -> Response:
        try:
            html_response = get(target_page)
        except RequestException:
            raise RequestException("Error in the request to the usernames page.")
        return html_response

    async def scrape_data(self) -> dict[str, list[tuple[str, float]]]:
        all_movie_data = {}
        for username_url in self.username_urls:
            target_page = os.path.join(RATINGS_PAGE, username_url, "films")
            all_movie_data[username_url] = await self.scrape_data_per_username_url(
                target_page
            )
        return all_movie_data

    async def scrape_data_per_username_url(
        self, username_url: str
    ) -> List[Tuple[str, float]]:
        response = await self.request_data(username_url)
        soup = BeautifulSoup(response.content, features="html.parser")

        try:
            pages_div = soup.find("div", class_="paginate-pages")
            n_pages = int(pages_div.find_all("li", class_="paginate-page")[-1].text)
        except AttributeError:
            logging.warning(
                "Number of pages not available. Setting the parameter to 1."
            )
            n_pages = 1

        all_movie_data: list[tuple[str, float]] = []
        print("Scraping data for user: %s", username_url)
        for id_page in range(1, 2 + 1):
            next_page: str = os.path.join(username_url, "page", str(id_page))
            print(f"Scraping data for page: {id_page} for username {username_url}")
            movie_data = await self.fetch_data(next_page)
            all_movie_data.extend(movie_data)
        return all_movie_data


    async def fetch_data(self, target_url: str) -> list[tuple[str, float]]:
        html_response = get(target_url)
        soup = BeautifulSoup(html_response.content, features="html.parser")
        movie_data = await self.get_data(soup)
        return movie_data

    async def get_data(self, soup: BeautifulSoup) -> list[tuple[str, float]]:
        movie_ratings = []
        poster_containers = soup.find_all("li", class_="poster-container")
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
