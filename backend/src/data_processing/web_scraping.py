from bs4 import BeautifulSoup
from requests import get
from typing import Tuple, List
import os


def request_movie_data(target_page: str) -> List[Tuple[str, float]]:
    html_response = get(target_page)
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

def convert_rating(rating: str) -> float:
    num_rating = float(len(rating)) if rating[-1] != "½" else len(rating) - 0.5
    return num_rating * 2

if __name__ == "__main__":
    target_page: str = "https://letterboxd.com/mattstouche/films/"
    print(request_movie_data(target_page))
