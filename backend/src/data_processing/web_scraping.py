from bs4 import BeautifulSoup
from requests import get
from typing import Tuple, List
import os

from requests.exceptions import RequestException


def request_usernames(usernames_page: str) -> List[str]:
    try:
        username_response = get(usernames_page)
    except:
        raise RequestException

    soup = BeautifulSoup(username_response.content, features="html.parser")
    pagination = soup.find("div", class_="pagination")
    is_next = True if pagination.find("a", class_="next") else False
    id_page: int = 1
    usernames: List[str] = []
    while is_next or id_page == 1:
        get_usernames(soup, usernames)
        id_page += 1
        next_page_postfix: str = f"page/{id_page}"
        next_page_address = os.path.join(username_page, next_page_postfix)
        username_response = get(next_page_address)
        soup = BeautifulSoup(username_response.content, features="html.parser")
        pagination = soup.find("div", class_="pagination")
        is_next = True if pagination.find("a", class_="next") else False
        break
    return usernames

def get_usernames(soup: BeautifulSoup, usernames: List[str]) -> None:
    usrs = soup.find_all("div", class_="person-summary")
    for usr in usrs:
        usr = usr.find("a", class_="name")["href"].split("/")[1]
        usernames.append(usr)
    return

def request_movie_data(target_page: str) -> List[Tuple[str, float]]:
    try:
        html_response = get(target_page)
    except:
        raise RequestException

    soup = BeautifulSoup(html_response.content, features="html.parser")
    pages_div = soup.find("div", class_="paginate-pages")
    n_pages = int(pages_div.find_all("li", class_="paginate-page")[-1].text)

    all_movie_data: List[Tuple[str, float]] = []
    for id_page in range(1, n_pages + 1):
        next_page: str = os.path.join(target_page, f"page/{id_page}")
        html_response = get(next_page)
        soup = BeautifulSoup(html_response.content, features="html.parser")
        get_movie_rating(soup, all_movie_data)
    return all_movie_data

def get_movie_rating(soup: BeautifulSoup, all_movie_data: List[Tuple[str, float]]) -> None:
    poster_containers = soup.find_all("li", class_="poster-container")
    for poster in poster_containers:
        movie_title = poster.find("img")["alt"]
        rating: str = poster.find("span", class_="rating").text
        num_rating = convert_rating(rating)
        all_movie_data.append((movie_title, num_rating))
    return

def convert_rating(rating: str) -> float:
    num_rating = float(len(rating)) if rating[-1] != "½" else len(rating) - 0.5
    return num_rating * 2

if __name__ == "__main__":
    target_page: str = "https://letterboxd.com/mattstouche/films/"
    username_page: str = "https://letterboxd.com/members/popular/this/week/"
    #print(request_movie_data(target_page))
    print(request_usernames(username_page))
