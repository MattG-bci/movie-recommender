from bs4 import BeautifulSoup

from abc import ABC, abstractmethod

from httpx import Response
from pydantic import BaseModel
from requests import get
from requests.exceptions import RequestException
from typing import Tuple, List

import os
import logging


class BaseWebScraper(ABC):
    @abstractmethod
    def request_data(self, url: str) -> List:
        raise NotImplementedError

    @abstractmethod
    def get_data(self, scraper) -> None:
        raise NotImplementedError


class UserScraper(BaseModel, BaseWebScraper):
    username_page: str

    def scrape_pages(self, start_page: int = 1, end_page: int = 11) -> list[str]:
        usernames = []
        for page in range(start_page, end_page):
            username_url = f"{self.username_page}/page/{page}/"
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


class RatingScraper(BaseWebScraper):
    def request_data(self, target_page: str) -> List[Tuple[str, float]]:
        try:
            html_response = get(target_page)
        except RequestException:
            return []

        soup = BeautifulSoup(html_response.content, features="html.parser")

        try:
            pages_div = soup.find("div", class_="paginate-pages")
            n_pages = int(pages_div.find_all("li", class_="paginate-page")[-1].text)
        except AttributeError:
            logging.warning(
                "Number of pages not available. Setting the parameter to 1."
            )
            n_pages = 1

        all_movie_data: List[Tuple[str, float]] = []
        for id_page in range(1, n_pages + 1):
            next_page: str = os.path.join(target_page, f"page/{id_page}")
            html_response = get(next_page)
            soup = BeautifulSoup(html_response.content, features="html.parser")
            self.get_data(soup, all_movie_data)
        return all_movie_data

    def get_data(
        self, soup: BeautifulSoup, all_movie_data: List[Tuple[str, float]]
    ) -> None:
        poster_containers = soup.find_all("li", class_="poster-container")
        for poster in poster_containers:
            movie_title = poster.find("img")["alt"]
            rating = poster.find("span", class_="rating")
            if not rating:
                continue
            num_rating = self.convert_rating(rating.text)
            all_movie_data.append((movie_title, num_rating))
        return

    @staticmethod
    def convert_rating(rating: str) -> float:
        num_rating = float(len(rating)) if rating[-1] != "½" else len(rating) - 0.5
        return num_rating * 2
