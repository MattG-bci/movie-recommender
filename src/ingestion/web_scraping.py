from bs4 import BeautifulSoup

from abc import ABC, abstractmethod
from requests import get
from requests.exceptions import RequestException
from typing import Tuple, List

import os
import logging


class WebScraper(ABC):
    @abstractmethod
    def request_data(self, url: str) -> List:
        raise NotImplementedError

    @abstractmethod
    def get_data(self, scraper, store) -> None:
        raise NotImplementedError


class UserScraper(WebScraper):
    def request_data(self, usernames_page: str) -> List[str]:
        try:
            username_response = get(usernames_page)
        except RequestException:
            return []

        soup = BeautifulSoup(username_response.content, features="html.parser")
        pagination = soup.find("div", class_="pagination")
        is_next = True if pagination.find("a", class_="next") else False
        id_page: int = 1
        usernames: List[str] = []
        while is_next or id_page == 1:
            self.get_data(soup, usernames)
            id_page += 1
            next_page_postfix: str = f"page/{id_page}"
            next_page_address = os.path.join(usernames_page, next_page_postfix)
            username_response = get(next_page_address)
            soup = BeautifulSoup(username_response.content, features="html.parser")
            pagination = soup.find("div", class_="pagination")
            is_next = True if pagination.find("a", class_="next") else False
        return usernames

    def get_data(self, soup: BeautifulSoup, usernames: List[str]) -> None:
        usrs = soup.find_all("div", class_="person-summary")
        for usr in usrs:
            usr = usr.find("a", class_="name")["href"].split("/")[1]
            usernames.append(usr)
        return


class RatingScraper(WebScraper):
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
            rating: str = poster.find("span", class_="rating")
            if not rating:
                continue
            num_rating = self.convert_rating(rating.text)
            all_movie_data.append((movie_title, num_rating))
        return

    def convert_rating(self, rating: str) -> float:
        num_rating = float(len(rating)) if rating[-1] != "½" else len(rating) - 0.5
        return num_rating * 2
